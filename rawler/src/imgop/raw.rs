// SPDX-License-Identifier: LGPL-2.1
// Copyright 2021 Daniel Vogelbacher <daniel@chaospixel.com>

use super::xyz::Illuminant;
use crate::imgop::Rect;
use crate::imgop::matrix::{multiply, normalize, pseudo_inverse};
use crate::imgop::xyz::SRGB_TO_XYZ_D65;
use crate::pixarray::{Color2D, RgbF32};
use crate::rawimage::{BlackLevel, RawPhotometricInterpretation, WhiteLevel};

use multiversion::multiversion;
use rayon::prelude::*;

/// Conversion matrix for a specific illuminant
#[derive(Debug)]
pub struct ColorMatrix {
  pub illuminant: Illuminant,
  pub matrix: [[f32; 3]; 4],
}

/// Parameters for raw image development
#[derive(Debug)]
pub struct DevelopParams {
  pub width: usize,
  pub height: usize,
  pub cpp: usize,
  pub color_matrices: Vec<ColorMatrix>,
  pub whitelevel: WhiteLevel,
  pub blacklevel: BlackLevel,
  //pub pattern: Option<BayerPattern>,
  pub photometric: RawPhotometricInterpretation,
  pub wb_coeff: [f32; 4],
  pub active_area: Option<Rect>,
  pub crop_area: Option<Rect>,
  //pub gamma: f32,
}

/// CLip only underflow values < 0.0
pub fn clip_negative<const N: usize>(pix: &[f32; N]) -> [f32; N] {
  pix.map(|p| if p < 0.0 { 0.0 } else { p })
}

/// Clip only overflow values > 1.0
pub fn clip_overflow<const N: usize>(pix: &[f32; N]) -> [f32; N] {
  pix.map(|p| if p > 1.0 { 1.0 } else { p })
}

/// Clip into range of 0.0 - 1.0
pub fn clip<const N: usize>(pix: &[f32; N]) -> [f32; N] {
  clip_range(pix, 0.0, 1.0)
}

/// Clip into range of lower and upper bound
pub fn clip_range<const N: usize>(pix: &[f32; N], lower: f32, upper: f32) -> [f32; N] {
  pix.map(|p| clip_value(p, lower, upper))
}

pub fn clip_value(p: f32, lower: f32, upper: f32) -> f32 {
  if p < lower {
    lower
  } else if p > upper {
    upper
  } else {
    p
  }
}

/// We ONLY clip black sun artifacts (negatives) and let highlights
/// remain completely unbounded (> 1.0) for the editor to recover.
pub fn clip_euclidean_norm_avg<const N: usize>(pix: &[f32; N]) -> [f32; N] {
  clip_negative(pix)
}

/// Correct data by blacklevel and whitelevel on CFA (bayer) data.
///
/// The output is between 0.0 .. 1.0.
///
/// This version is optimized vor vectorization, so please check
/// modifications on godbolt before committing.
#[multiversion(targets("x86_64+avx+avx2", "x86+sse", "aarch64+neon"))]
fn correct_blacklevel_channels<const CH: usize>(raw: &mut [f32], blacklevel: &[f32; CH], whitelevel: &[f32; CH]) {
  // max value can be pre-computed for all channels.
  let mut max = *whitelevel;
  max.iter_mut().enumerate().for_each(|(i, x)| *x -= blacklevel[i]);

  let clip = |v: f32| {
    if v.is_sign_negative() { 0.0 } else { v }
  };
  if CH == 1 {
    let max = max[0];
    let blacklevel = blacklevel[0];
    raw.iter_mut().for_each(|p| *p = clip(*p - blacklevel) / max);
  } else {
    raw.chunks_exact_mut(CH).for_each(|block| {
      for i in 0..CH {
        block[i] = clip(block[i] - blacklevel[i]) / max[i];
      }
    });
  }
}

/// Correct pixel data with `cpp` samples per pixel by a black level pattern
/// and the white level.
///
/// The black level pattern is `black.width` x `black.height` pixels (DNG
/// BlackLevelRepeatDim), repeated from the top-left pixel, with a value for
/// each sample or one for all samples. A white level has a value for each
/// sample or one for all. 1.0 is the white level minus the highest black
/// level of the sample, values below black are 0.0.
///
/// Adapted from dnglab's `RescaleRows` and `BlackLevel::max_level`
/// (rawler/src/imgop/blacklevel.rs @ 0f645aa6ed, "Map raw values to linear
/// values by the DNG black level model"), without the BlackLevelDeltaH/V
/// differences this fork does not read.
pub fn correct_blacklevel_pattern(raw: &mut [f32], width: usize, cpp: usize, black: &BlackLevel, white: &WhiteLevel) -> Result<(), String> {
  if black.cpp != cpp && black.cpp != 1 {
    return Err(format!("Black level has {} samples per pixel, image has {}", black.cpp, cpp));
  }
  if white.0.len() != cpp && white.0.len() != 1 {
    return Err(format!("White level has {} values, image has {} samples per pixel", white.0.len(), cpp));
  }
  if black.levels.len() != black.sample_count() || black.levels.is_empty() || width == 0 || cpp == 0 {
    return Err(format!(
      "Black level pattern has {} values, expected {}",
      black.levels.len(),
      black.sample_count()
    ));
  }
  let level = |row: usize, col: usize, sample: usize| {
    black.levels[((row % black.height) * black.width + col % black.width) * black.cpp + sample.min(black.cpp - 1)].as_f32()
  };
  // The linear value 1.0 is the white level minus the maximum black level of each sample
  let range: Vec<f32> = (0..cpp)
    .map(|sample| {
      let max_black = (0..black.height)
        .flat_map(|row| (0..black.width).map(move |col| (row, col)))
        .map(|(row, col)| level(row, col, sample))
        .fold(f32::NEG_INFINITY, f32::max);
      white.0[sample.min(white.0.len() - 1)] as f32 - max_black
    })
    .collect();
  if let Some(range) = range.iter().find(|range| !range.is_finite() || **range <= 0.0) {
    return Err(format!("White level must be above the maximum black level, the difference is {}", range));
  }
  // One row of black levels for each row of the pattern, the width of the image
  let row_len = width * cpp;
  let black_rows: Vec<Vec<f32>> = (0..black.height)
    .map(|row| (0..row_len).map(|i| level(row, i / cpp, i % cpp)).collect())
    .collect();
  let scale: Vec<f32> = (0..row_len).map(|i| 1.0 / range[i % cpp]).collect();
  raw.par_chunks_mut(row_len).enumerate().for_each(|(row, samples)| {
    for ((v, black), scale) in samples.iter_mut().zip(&black_rows[row % black.height]).zip(&scale) {
      *v = (*v - black).max(0.0) * scale;
    }
  });
  Ok(())
}

/// Correct data by blacklevel and whitelevel on CFA (bayer) data.
///
/// White is 1.0. A sample below black is 0.0 with `clip_black`, and negative
/// without it, for a demosaic that averages the noise there before it is clipped.
///
/// This version is optimized vor vectorization, so please check
/// modifications on godbolt before committing.
#[multiversion(targets("x86_64+avx+avx2", "x86+sse", "aarch64+neon"))]
pub fn correct_blacklevel_cfa(raw: &mut [f32], width: usize, _height: usize, blacklevel: &[f32; 4], whitelevel: &[f32; 4], clip_black: bool) {
  //assert_eq!(width % 2, 0, "width is {}", width);
  //assert_eq!(height % 2, 0, "height is {}", height);
  // max value can be pre-computed for all channels.
  let max = [
    whitelevel[0] - blacklevel[0],
    whitelevel[1] - blacklevel[1],
    whitelevel[2] - blacklevel[2],
    whitelevel[3] - blacklevel[3],
  ];
  let clip = |v: f32| {
    if clip_black && v.is_sign_negative() { 0.0 } else { v }
  };
  // Process two bayer lines at once.
  raw.par_chunks_exact_mut(width * 2).for_each(|lines| {
    // It's bayer data, so we have two lines for sure.
    let (line0, line1) = lines.split_at_mut(width);
    //line0.array_chunks_mut::<2>().zip(line1.array_chunks_mut::<2>()).for_each(|(a, b)| {
    line0.chunks_exact_mut(2).zip(line1.chunks_exact_mut(2)).for_each(|(a, b)| {
      a[0] = clip(a[0] - blacklevel[0]) / max[0];
      a[1] = clip(a[1] - blacklevel[1]) / max[1];
      b[0] = clip(b[0] - blacklevel[2]) / max[2];
      b[1] = clip(b[1] - blacklevel[3]) / max[3];
    });
  });
}

#[multiversion(targets("x86_64+avx+avx2", "x86+sse", "aarch64+neon"))]
pub(crate) fn map_3ch_to_rgb(src: &Color2D<f32, 3>, wb_coeff: &[f32; 4], xyz2cam: [[f32; 3]; 4]) -> RgbF32 {
  let rgb2cam = normalize(multiply(&xyz2cam, &SRGB_TO_XYZ_D65));
  let cam2rgb = pseudo_inverse(rgb2cam);

  let mut out = Vec::with_capacity(src.data.len());

  src
    .pixels()
    .par_iter()
    .map(|pix| {
      // We apply wb coeffs on the fly
      let r = pix[0] * wb_coeff[0];
      let g = pix[1] * wb_coeff[1];
      let b = pix[2] * wb_coeff[2];
      let srgb = [
        cam2rgb[0][0] * r + cam2rgb[0][1] * g + cam2rgb[0][2] * b,
        cam2rgb[1][0] * r + cam2rgb[1][1] * g + cam2rgb[1][2] * b,
        cam2rgb[2][0] * r + cam2rgb[2][1] * g + cam2rgb[2][2] * b,
      ];
      clip_euclidean_norm_avg(&srgb)
    })
    .collect_into_vec(&mut out);

  RgbF32::new_with(out, src.width, src.height)
}

#[multiversion(targets("x86_64+avx+avx2", "x86+sse", "aarch64+neon"))]
pub(crate) fn map_4ch_to_rgb(src: &Color2D<f32, 4>, wb_coeff: &[f32; 4], xyz2cam: [[f32; 3]; 4]) -> RgbF32 {
  let rgb2cam = normalize(multiply(&xyz2cam, &SRGB_TO_XYZ_D65));
  let cam2rgb = pseudo_inverse(rgb2cam);

  let mut out = Vec::with_capacity(src.data.len());

  src
    .pixels()
    .par_iter()
    .map(|pix| {
      // We apply wb coeffs on the fly
      let ch0 = pix[0] * wb_coeff[0];
      let ch1 = pix[1] * wb_coeff[1];
      let ch2 = pix[2] * wb_coeff[2];
      let ch3 = pix[3] * wb_coeff[3];
      let srgb = [
        cam2rgb[0][0] * ch0 + cam2rgb[0][1] * ch1 + cam2rgb[0][2] * ch2 + cam2rgb[0][3] * ch3,
        cam2rgb[1][0] * ch0 + cam2rgb[1][1] * ch1 + cam2rgb[1][2] * ch2 + cam2rgb[1][3] * ch3,
        cam2rgb[2][0] * ch0 + cam2rgb[2][1] * ch1 + cam2rgb[2][2] * ch2 + cam2rgb[2][3] * ch3,
      ];
      clip_euclidean_norm_avg(&srgb)
    })
    .collect_into_vec(&mut out);

  RgbF32::new_with(out, src.width, src.height)
}

/// Collect iterator into array
pub fn collect_array<T, I, const N: usize>(itr: I) -> [T; N]
where
  T: Default + Copy,
  I: IntoIterator<Item = T>,
{
  let mut res = [T::default(); N];
  for (it, elem) in res.iter_mut().zip(itr) {
    *it = elem
  }

  res
}

/// Calculate the multiplicative invert of an array
/// (sum of each row equals to 1.0)
pub fn mul_invert_array<const N: usize>(a: &[f32; N]) -> [f32; N] {
  let mut b = [f32::default(); N];
  b.iter_mut().zip(a.iter()).for_each(|(x, y)| *x = 1.0 / y);
  b
}

pub fn rotate_90(src: &[u16], dst: &mut [u16], width: usize, height: usize) {
  let dst = &mut dst[..src.len()]; // optimize len hints for compiler
  let owidth = height;
  for (row, line) in src.chunks_exact(width).enumerate() {
    for (col, pix) in line.iter().enumerate() {
      let orow = col;
      let ocol = (owidth - 1) - row; // inverse
      dst[orow * owidth + ocol] = *pix;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// LinearRaw with a 2x2 black level pattern of three samples (12 values, as
  /// in phone DNGs with BlackLevelRepeatDim 2x2) and one white level per sample
  #[test]
  fn black_level_pattern_with_white_level_per_sample() -> Result<(), String> {
    let black = BlackLevel::new(&[10_u16, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33], 2, 2, 3);
    let white = WhiteLevel::new(vec![1013_u32, 2023, 4033]);
    // 3x2 pixels: the pattern repeats from the top-left pixel
    let (width, cpp) = (3, 3);
    let mut raw: Vec<f32> = vec![
      10.0, 20.0, 30.0, 511.0, 1021.0, 2031.0, 1010.0, 2020.0, 4030.0, //
      12.0, 22.0, 0.0, 1013.0, 2023.0, 4033.0, 512.0, 1022.0, 2032.0,
    ];
    correct_blacklevel_pattern(&mut raw, width, cpp, &black, &white)?;
    // 1.0 is the white level minus the highest black level of the sample: 1000, 2000, 4000
    let expected = [
      0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, //
      0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5,
    ];
    for (v, e) in raw.iter().zip(expected) {
      assert!((v - e).abs() < 1e-6, "{:?} != {:?}", raw, expected);
    }
    Ok(())
  }

  #[test]
  fn black_level_pattern_with_one_value() -> Result<(), String> {
    let mut raw = vec![16383.0_f32, 4279.0, 0.0];
    correct_blacklevel_pattern(&mut raw, 1, 3, &BlackLevel::new(&[0_u16], 1, 1, 1), &WhiteLevel::new(vec![16383_u32]))?;
    assert_eq!(raw, vec![1.0, 4279.0 / 16383.0, 0.0]);
    Ok(())
  }

  #[test]
  fn black_level_pattern_rejects_levels_that_do_not_fit() {
    let mut raw = vec![0.0_f32; 6];
    let black = BlackLevel::new(&[0_u16, 0], 1, 1, 2);
    assert!(correct_blacklevel_pattern(&mut raw, 2, 3, &black, &WhiteLevel::new(vec![100_u32])).is_err());
    let black = BlackLevel::new(&[0_u16], 1, 1, 1);
    assert!(correct_blacklevel_pattern(&mut raw, 2, 3, &black, &WhiteLevel::new(vec![100_u32, 100])).is_err());
    let black = BlackLevel::new(&[100_u16], 1, 1, 1);
    assert!(correct_blacklevel_pattern(&mut raw, 2, 3, &black, &WhiteLevel::new(vec![100_u32])).is_err());
  }
}
