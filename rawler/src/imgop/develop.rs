use std::io;

use image::{DynamicImage, ImageBuffer};

use crate::{
  RawImage,
  decoders::RawMetadata,
  formats::tiff::{DirectoryWriter, TiffWriter},
  pixarray::{Color2D, PixF32},
  rawimage::RawPhotometricInterpretation,
  tags::{ExifTag, TiffCommonTag},
};

use super::{
  Dim2, Rect, convert_from_f32_scaled_u16,
  raw::{clip_negative, map_3ch_to_rgb, map_4ch_to_rgb},
  sensor::bayer::{
    bilinear::Bilinear4Channel, ppg::PPGDemosaic, superpixel::{Superpixel4Channel, SuperpixelQuarterRes3Channel}, Demosaic,
  },
  sensor::xtrans::demosaic::{XTransDemosaic, XTransSuperpixelDemosaic},
  xyz::Illuminant,
};

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ProcessingStep {
  Rescale,
  Demosaic,
  CropActiveArea,
  WhiteBalance,
  Calibrate,
  CropDefault,
  SRgb,
}

/// The demosaicing algorithm to use.
#[derive(PartialEq, Eq, Debug, Clone, Copy, Default)]
pub enum DemosaicAlgorithm {
  /// High-quality demosaicing (PPG for Bayer, Full-Res for X-Trans).
  #[default]
  Quality,
  /// High-speed demosaicing using a superpixel algorithm (e.g. for thumbnails).
  /// This reduces image dimensions by a factor of four (quarter width and height).
  Speed,
}

pub struct RawDevelopBuilder {}

#[derive(Clone)]
pub enum Intermediate {
  Monochrome(PixF32),
  ThreeColor(Color2D<f32, 3>),
  FourColor(Color2D<f32, 4>),
}

impl Intermediate {
  pub fn dim(&self) -> Dim2 {
    match self {
      Intermediate::Monochrome(pixels) => pixels.dim(),
      Intermediate::ThreeColor(pixels) => pixels.dim(),
      Intermediate::FourColor(pixels) => pixels.dim(),
    }
  }

  pub fn rect(&self) -> Rect {
    match self {
      Intermediate::Monochrome(pixels) => pixels.rect(),
      Intermediate::ThreeColor(pixels) => pixels.rect(),
      Intermediate::FourColor(pixels) => pixels.rect(),
    }
  }

  pub fn to_dynamic_image(self) -> Option<DynamicImage> {
    Some(match self {
      Intermediate::Monochrome(pixels) => {
        let data = convert_from_f32_scaled_u16(&pixels.data, 0, u16::MAX);
        DynamicImage::ImageLuma16(ImageBuffer::from_raw(pixels.dim().w as u32, pixels.dim().h as u32, data)?)
      }
      Intermediate::ThreeColor(pixels) => {
        let data = convert_from_f32_scaled_u16(&pixels.flatten(), 0, u16::MAX);
        DynamicImage::ImageRgb16(ImageBuffer::from_raw(pixels.dim().w as u32, pixels.dim().h as u32, data)?)
      }
      Intermediate::FourColor(pixels) => {
        let data = convert_from_f32_scaled_u16(&pixels.flatten(), 0, u16::MAX);
        DynamicImage::ImageRgba16(ImageBuffer::from_raw(pixels.dim().w as u32, pixels.dim().h as u32, data)?)
      }
    })
  }
}

#[derive(Clone)]
pub struct RawDevelop {
  pub steps: Vec<ProcessingStep>,
  pub demosaic_algorithm: DemosaicAlgorithm,
}

impl Default for RawDevelop {
  fn default() -> Self {
    Self {
      steps: vec![
        ProcessingStep::Rescale,
        ProcessingStep::Demosaic,
        ProcessingStep::CropActiveArea,
        ProcessingStep::WhiteBalance,
        ProcessingStep::Calibrate,
        ProcessingStep::CropDefault,
        ProcessingStep::SRgb,
      ],
      demosaic_algorithm: DemosaicAlgorithm::default(),
    }
  }
}

impl RawDevelop {
  /*
  pub fn linearize(rawimage: &RawImage) -> crate::Result<RgbF32> {
    todo!()
  }

  pub fn develop_monochrome_image(&self, rawimage: &RawImage) -> crate::Result<PixF32> {
    todo!()
  }

  pub fn develop_rgb_image(&self, rawimage: &RawImage) -> crate::Result<RgbF32> {
    todo!()
  }
   */

  /// Develop raw image and write result into TIFF.
  /// If demosaic is disabled or camera raw is monochrome, the TIFF
  /// has only one color channel.
  pub fn develop_intermediate(&self, rawimage: &RawImage) -> crate::Result<Intermediate> {
    let mut rawimage = rawimage.clone();
    if self.steps.contains(&ProcessingStep::Rescale) {
      rawimage.apply_scaling()?;
    }

    let mut intermediate = match rawimage.cpp {
      1 => Intermediate::Monochrome(PixF32::new_with(rawimage.data.as_f32().into_owned(), rawimage.width, rawimage.height)),
      3 => Intermediate::ThreeColor(Color2D::<f32, 3>::new_with(
        rawimage.data.as_f32().chunks_exact(3).map(|x| [x[0], x[1], x[2]]).collect(),
        rawimage.width,
        rawimage.height,
      )),
      4 => Intermediate::FourColor(Color2D::<f32, 4>::new_with(
        rawimage.data.as_f32().chunks_exact(4).map(|x| [x[0], x[1], x[2], x[3]]).collect(),
        rawimage.width,
        rawimage.height,
      )),
      _ => todo!(),
    };

    if self.steps.contains(&ProcessingStep::Demosaic) {
      intermediate = match &rawimage.photometric {
        RawPhotometricInterpretation::Cfa(config) => {
          log::info!(
              "Demosaicing check for '{} {}': CFA name='{}', width={}, height={}, is_rgb={}",
              rawimage.clean_make,
              rawimage.clean_model,
              config.cfa.name,
              config.cfa.width,
              config.cfa.height,
              config.cfa.is_rgb()
          );
          if let Intermediate::Monochrome(ref pixels) = intermediate {
            let roi = if self.steps.contains(&ProcessingStep::CropActiveArea) {
              rawimage.active_area.unwrap_or(pixels.rect())
            } else {
              pixels.rect()
            };
            if config.cfa.is_rgb() {
              // Check if this is an X-Trans sensor (6x6 pattern) 
              if config.cfa.is_xtrans() {
                log::info!("X-Trans pattern (6x6) detected. Applying X-Trans demosaicing ({:?}).", self.demosaic_algorithm);
                match self.demosaic_algorithm {
                  DemosaicAlgorithm::Quality => {
                    let xtrans_demosaic = XTransDemosaic::new();
                    Intermediate::ThreeColor(xtrans_demosaic.demosaic(pixels, &config.cfa, &config.colors, roi))
                  }
                  DemosaicAlgorithm::Speed => {
                    let xtrans_demosaic = XTransSuperpixelDemosaic::new();
                    Intermediate::ThreeColor(xtrans_demosaic.demosaic(pixels, &config.cfa, &config.colors, roi))
                  }
                }
              } else {
                  log::info!("RGB Bayer-like pattern detected. Applying Bayer demosaicing.");
                  match self.demosaic_algorithm {
                      DemosaicAlgorithm::Quality => {
                          let ppg = PPGDemosaic::new();
                          Intermediate::ThreeColor(ppg.demosaic(pixels, &config.cfa, &config.colors, roi))
                      }
                      DemosaicAlgorithm::Speed => {
                          let superpixel = SuperpixelQuarterRes3Channel::new();
                          Intermediate::ThreeColor(superpixel.demosaic(pixels, &config.cfa, &config.colors, roi))
                      }
                  }
              }
            } else if config.cfa.unique_colors() == 4 {
                log::info!("4-Color pattern detected. Applying 4-channel demosaicing.");
                match self.demosaic_algorithm {
                    DemosaicAlgorithm::Quality => {
                        let linear = Bilinear4Channel::new();
                        Intermediate::FourColor(linear.demosaic(&pixels, &config.cfa, &config.colors, roi))
                    }
                    DemosaicAlgorithm::Speed => {
                        let superpixel = Superpixel4Channel::new();
                        Intermediate::FourColor(superpixel.demosaic(&pixels, &config.cfa, &config.colors, roi))
                    }
                }
            } else {
                log::warn!("Unsupported CFA pattern '{}' for demosaicing. Passing through without demosaicing.", config.cfa.name);
                Intermediate::Monochrome(pixels.clone())
            }
          } else {
            intermediate
          }
        }
        _ => intermediate,
      };
    }

    // Noise below black is clipped here, once, not sample by sample: an interpolated
    // sample averages its neighbours' noise, so a dark noisy area's mean rises less, and
    // white balance scales less of that rise into a colour cast. RAWmakase 4d3f6f8c:
    // crates/rawmakase-decode/src/demosaic.rs. X-Trans still clips each sample first.
    match &mut intermediate {
      Intermediate::Monochrome(pixels) => pixels.for_each(|v| v.max(0.0)),
      Intermediate::ThreeColor(pixels) => pixels.for_each(|p| clip_negative(&p)),
      Intermediate::FourColor(pixels) => pixels.for_each(|p| clip_negative(&p)),
    };

    if self.steps.contains(&ProcessingStep::Calibrate) {
      let found_matrix = rawimage
        .color_matrix
        .iter()
        .find(|(illuminant, _m)| **illuminant == Illuminant::D65)
        .or_else(|| rawimage.color_matrix.iter().next());

      if let Some((_illuminant, color_matrix)) = found_matrix {
        // Safety check: ensure matrix is valid length
        if color_matrix.len() % 3 == 0 {
            let mut xyz2cam: [[f32; 3]; 4] = [[0.0; 3]; 4];
            let components = color_matrix.len() / 3;
            for i in 0..components {
              for j in 0..3 {
                xyz2cam[i][j] = color_matrix[i * 3 + j];
              }
            }

            let mut wb = if rawimage.wb_coeffs[0].is_nan() {
              [1.0, 1.0, 1.0, 1.0]
            } else {
              rawimage.wb_coeffs
            };
            if !self.steps.contains(&ProcessingStep::WhiteBalance) {
              wb = [1.0, 1.0, 1.0, 1.0];
            }

            log::info!("Applying calibration with wb: {:?}, xyz2cam: {:?}", wb, xyz2cam);

            intermediate = match intermediate {
              Intermediate::Monochrome(_) => intermediate,
              Intermediate::ThreeColor(pixels) => Intermediate::ThreeColor(map_3ch_to_rgb(&pixels, &wb, xyz2cam)),
              Intermediate::FourColor(pixels) => Intermediate::ThreeColor(map_4ch_to_rgb(&pixels, &wb, xyz2cam)),
            };
        } else {
            log::warn!("Color matrix found but has invalid length. Skipping calibration.");
        }
      } else {
        log::warn!("Illuminant matrix D65 not found and no fallback available. Skipping calibration.");
      }
    }

    if self.steps.contains(&ProcessingStep::CropDefault) {
      if let Some(mut crop) = rawimage.crop_area {
        if self.steps.contains(&ProcessingStep::Demosaic) && self.steps.contains(&ProcessingStep::CropActiveArea) {
          if let Some(active_area) = rawimage.active_area {
            let intersection = crop.intersection(&active_area);
            crop = intersection.adapt(&active_area);
          }
        }
        let original_width = rawimage.active_area.map(|area| area.d.w).unwrap_or(rawimage.dim().w);
        if original_width > 0 {
            let scale_factor = intermediate.dim().w as f32 / original_width as f32;
            if (scale_factor - 1.0).abs() > 1e-6 {
                crop.scale(scale_factor);
            }
        }
        if !crop.is_empty() && crop.d != intermediate.dim() {
          log::info!("crop: {:?}, intermediate dim: {:?}, rawimage: {:?}", crop, intermediate.dim(), rawimage.dim());
          intermediate = match intermediate {
            Intermediate::Monochrome(pixels) => Intermediate::Monochrome(pixels.crop(crop)),
            Intermediate::ThreeColor(pixels) => Intermediate::ThreeColor(pixels.crop(crop)),
            Intermediate::FourColor(pixels) => Intermediate::FourColor(pixels.crop(crop)),
          };
        }
      }
    }

    if self.steps.contains(&ProcessingStep::SRgb) {
      match &mut intermediate {
        Intermediate::Monochrome(pixels) => pixels.for_each(super::srgb::srgb_apply_gamma),
        Intermediate::ThreeColor(pixels) => pixels.for_each(super::srgb::srgb_apply_gamma_n),
        Intermediate::FourColor(pixels) => pixels.for_each(super::srgb::srgb_apply_gamma_n),
      };
    }

    Ok(intermediate)
  }

  /// Develop raw image and write result into TIFF.
  /// If demosaic is disabled or camera raw is monochrome, the TIFF
  /// has only one color channel.
  pub fn develop<W>(&self, rawimage: &RawImage, md: &RawMetadata, writer: W) -> crate::Result<()>
  where
    W: io::Write + io::Seek,
  {
    let intermediate = self.develop_intermediate(rawimage)?;

    let mut tiff = TiffWriter::new(writer)?;
    let mut root_ifd = DirectoryWriter::new();
    let mut exif_ifd = DirectoryWriter::new();

    // Add EXIF version 0220
    exif_ifd.add_tag_undefined(ExifTag::ExifVersion, vec![48, 50, 50, 48]);

    md.write_exif_tags(&mut tiff, &mut root_ifd, &mut exif_ifd)?;
    root_ifd.add_tag(TiffCommonTag::Make, rawimage.clean_make.as_str());
    root_ifd.add_tag(TiffCommonTag::Model, rawimage.clean_model.as_str());

    let exif_offset = exif_ifd.build(&mut tiff)?;

    root_ifd.add_tag(TiffCommonTag::ExifIFDPointer, exif_offset);

    match intermediate {
      Intermediate::Monochrome(pixels) => {
        let data = convert_from_f32_scaled_u16(&pixels.data, 0, u16::MAX);
        let (strip_rows, strips) = tiff.write_strips_lzw(&data, 1, pixels.dim(), 0)?;
        let strip_offsets: Vec<u32> = strips.iter().map(|(offset, _)| *offset).collect();
        let strip_bytes: Vec<u32> = strips.iter().map(|(_, bytes)| *bytes).collect();
        root_ifd.add_tag(TiffCommonTag::Compression, 5);
        root_ifd.add_tag(TiffCommonTag::Predictor, 1);
        root_ifd.add_tag(TiffCommonTag::StripOffsets, &strip_offsets);
        root_ifd.add_tag(TiffCommonTag::StripByteCounts, &strip_bytes);
        root_ifd.add_tag(TiffCommonTag::BitsPerSample, [16_u16]);
        root_ifd.add_tag(TiffCommonTag::SamplesPerPixel, [1_u16]);
        root_ifd.add_tag(TiffCommonTag::PhotometricInt, [1_u16]);
        root_ifd.add_tag(TiffCommonTag::RowsPerStrip, strip_rows);
        root_ifd.add_tag(TiffCommonTag::ImageWidth, pixels.width as u16);
        root_ifd.add_tag(TiffCommonTag::ImageLength, pixels.height as u16);
      }
      Intermediate::ThreeColor(pixels) => {
        let data = convert_from_f32_scaled_u16(&pixels.flatten(), 0, u16::MAX);
        let (strip_rows, strips) = tiff.write_strips_lzw(&data, 3, pixels.dim(), 0)?;
        let strip_offsets: Vec<u32> = strips.iter().map(|(offset, _)| *offset).collect();
        let strip_bytes: Vec<u32> = strips.iter().map(|(_, bytes)| *bytes).collect();
        root_ifd.add_tag(TiffCommonTag::Compression, 5);
        root_ifd.add_tag(TiffCommonTag::Predictor, 1);
        root_ifd.add_tag(TiffCommonTag::StripOffsets, &strip_offsets);
        root_ifd.add_tag(TiffCommonTag::StripByteCounts, &strip_bytes);
        root_ifd.add_tag(TiffCommonTag::BitsPerSample, [16_u16, 16, 16]);
        root_ifd.add_tag(TiffCommonTag::SamplesPerPixel, [3_u16]);
        root_ifd.add_tag(TiffCommonTag::PhotometricInt, [2_u16]);
        root_ifd.add_tag(TiffCommonTag::RowsPerStrip, strip_rows);
        root_ifd.add_tag(TiffCommonTag::ImageWidth, pixels.width as u16);
        root_ifd.add_tag(TiffCommonTag::ImageLength, pixels.height as u16);
      }
      Intermediate::FourColor(pixels) => {
        let data = convert_from_f32_scaled_u16(&pixels.flatten(), 0, u16::MAX);
        let (strip_rows, strips) = tiff.write_strips_lzw(&data, 4, pixels.dim(), 0)?;
        let strip_offsets: Vec<u32> = strips.iter().map(|(offset, _)| *offset).collect();
        let strip_bytes: Vec<u32> = strips.iter().map(|(_, bytes)| *bytes).collect();
        root_ifd.add_tag(TiffCommonTag::Compression, 5);
        root_ifd.add_tag(TiffCommonTag::Predictor, 1);
        root_ifd.add_tag(TiffCommonTag::StripOffsets, &strip_offsets);
        root_ifd.add_tag(TiffCommonTag::StripByteCounts, &strip_bytes);
        root_ifd.add_tag(TiffCommonTag::BitsPerSample, [16_u16, 16, 16, 16]); // Extra-channel, even if PhotometricInt is RGB!
        root_ifd.add_tag(TiffCommonTag::SamplesPerPixel, [4_u16]);
        root_ifd.add_tag(TiffCommonTag::PhotometricInt, [2_u16]);
        root_ifd.add_tag(TiffCommonTag::RowsPerStrip, strip_rows);
        root_ifd.add_tag(TiffCommonTag::ImageWidth, pixels.width as u16);
        root_ifd.add_tag(TiffCommonTag::ImageLength, pixels.height as u16);
      }
    }

    tiff.build(root_ifd)?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    CFA, RawImageData,
    cfa::PlaneColor,
    decoders::Camera,
    imgop::sensor::bayer::superpixel::Superpixel3Channel,
    rawimage::{BlackLevel, CFAConfig, WhiteLevel},
  };

  const BLACK: u16 = 1000;
  const WHITE: u32 = 16000;
  const X_TRANS: &str = "RBGBRGGGRGGBGGBGGRBRGRBGGGBGGRGGRGGB";

  /// `samples` as a mosaic of one sample a pixel, its levels `BLACK` and `WHITE`.
  fn mosaic(pattern: &str, planes: &str, samples: Vec<u16>, width: usize, height: usize) -> RawImage {
    let cfa = CFA::new(pattern);
    let colors = PlaneColor::new(planes);
    RawImage::new_with_data(
      Camera { cfa: cfa.clone(), plane_color: colors.clone(), ..Default::default() },
      RawImageData::Integer(samples),
      width,
      height,
      1,
      [1.0; 4],
      RawPhotometricInterpretation::Cfa(CFAConfig::new(&cfa, &colors)),
      Some(BlackLevel::new(&[BLACK], 1, 1, 1)),
      Some(WhiteLevel::new(vec![WHITE])),
      false,
    )
  }

  /// `level` counts above black, plus and minus `noise` in turn, so that each 4 x 4 block
  /// holds every colour's samples half above and half below `level`.
  fn square_noise(width: usize, height: usize, level: u16, noise: u16) -> Vec<u16> {
    (0..height)
      .flat_map(|y| (0..width).map(move |x| if (x / 2 + y / 2) % 2 == 0 { BLACK + level + noise } else { BLACK + level - noise }))
      .collect()
  }

  /// A repeatable spread of about `noise` around `level` counts above black.
  fn random_noise(width: usize, height: usize, level: u16, noise: u16) -> Vec<u16> {
    let mut state = 12345_u32;
    let mut uniform = move || {
      state = state.wrapping_mul(1664525).wrapping_add(1013904223);
      (state >> 16) as f32 / 65536.0
    };
    (0..width * height)
      .map(|_| {
        let sum: f32 = (0..12).map(|_| uniform()).sum();
        (BLACK as f32 + level as f32 + (sum - 6.0) * noise as f32) as u16
      })
      .collect()
  }

  fn three_colours(intermediate: Intermediate) -> Vec<[f32; 3]> {
    match intermediate {
      Intermediate::ThreeColor(pixels) => pixels.data,
      _ => panic!("expected three colours a pixel"),
    }
  }

  fn demosaiced(raw: &RawImage, demosaic_algorithm: DemosaicAlgorithm) -> Vec<[f32; 3]> {
    let develop = RawDevelop {
      steps: vec![ProcessingStep::Rescale, ProcessingStep::Demosaic, ProcessingStep::CropActiveArea],
      demosaic_algorithm,
    };
    three_colours(develop.develop_intermediate(raw).unwrap())
  }

  fn scaled(raw: &RawImage) -> Vec<f32> {
    let mut raw = raw.clone();
    raw.apply_scaling().unwrap();
    raw.data.as_f32().into_owned()
  }

  #[test]
  fn scaling_keeps_a_bayer_sample_below_black_negative() {
    let raw = mosaic("RGGB", "RGB", [BLACK - 100, BLACK, BLACK + 100, BLACK].repeat(2), 4, 2);
    let span = (WHITE - BLACK as u32) as f32;
    assert_eq!(scaled(&raw)[..3], [-100.0 / span, 0.0, 100.0 / span]);
  }

  #[test]
  fn scaling_clips_an_x_trans_sample_at_black() {
    let raw = mosaic(X_TRANS, "RGB", square_noise(12, 12, 0, 100), 12, 12);
    assert!(scaled(&raw).iter().all(|v| *v >= 0.0));
  }

  #[test]
  fn scaling_clips_a_four_colour_sample_at_black() {
    let raw = mosaic("RGEB", "RGEB", square_noise(8, 8, 0, 100), 8, 8);
    assert!(scaled(&raw).iter().all(|v| *v >= 0.0));
  }

  /// Each block's samples average 10 counts above black, so that is what comes
  /// out, not the 25 they average once each is clipped at black.
  #[test]
  fn the_speed_demosaic_averages_samples_below_black_before_clipping() {
    let raw = mosaic("RGGB", "RGB", square_noise(16, 16, 10, 40), 16, 16);
    let want = 10.0 / (WHITE - BLACK as u32) as f32;
    for pixel in demosaiced(&raw, DemosaicAlgorithm::Speed) {
      for value in pixel {
        assert!((value - want).abs() < want * 1e-3, "{value} is not {want}");
      }
    }
  }

  /// An area darker than its noise comes out nearer its true level than its
  /// samples do once each is clipped at black.
  #[test]
  fn the_quality_demosaic_interpolates_samples_below_black_before_clipping() {
    let (width, height, level) = (64, 64, 20);
    let samples = random_noise(width, height, level, 60);
    let clipped_first: f32 =
      samples.iter().map(|&v| v.saturating_sub(BLACK) as f32).sum::<f32>() / samples.len() as f32 / (WHITE - BLACK as u32) as f32;
    let raw = mosaic("RGGB", "RGB", samples, width, height);
    let out = demosaiced(&raw, DemosaicAlgorithm::Quality);
    let mean: f32 = out.iter().flatten().sum::<f32>() / (out.len() * 3) as f32;
    assert!(mean < clipped_first * 0.95, "mean {mean}, clipped first {clipped_first}");
  }

  #[test]
  fn nothing_below_black_reaches_white_balance() {
    let raw = mosaic("RGGB", "RGB", random_noise(64, 64, 20, 60), 64, 64);
    for algorithm in [DemosaicAlgorithm::Quality, DemosaicAlgorithm::Speed] {
      assert!(demosaiced(&raw, algorithm).iter().flatten().all(|v| *v >= 0.0));
    }
  }

  /// Red and blue are single samples at this size and green averages two, so clipping
  /// after averaging would lower green alone and leave a dark area more magenta.
  #[test]
  fn the_half_size_superpixel_clips_each_sample_first() {
    let cfa = CFA::new("RGGB");
    let pixels = PixF32::new_with(vec![-0.25, -0.5, 0.75, 0.25], 2, 2);
    let half = Superpixel3Channel::new().demosaic(&pixels, &cfa, &PlaneColor::new("RGB"), pixels.rect());
    assert_eq!(half.data, [[0.0, 0.375, 0.25]]);
  }

  /// A picture demosaiced elsewhere, as a half-size develop's is, is clipped the same way.
  #[test]
  fn three_colours_from_elsewhere_are_clipped_before_white_balance() {
    let raw = RawImage::new_with_data(
      Camera::default(),
      RawImageData::Float(vec![-0.25, 0.5, 0.75, 0.25, -0.5, 0.0]),
      6,
      1,
      3,
      [1.0; 4],
      RawPhotometricInterpretation::LinearRaw,
      None,
      Some(WhiteLevel::new(vec![1, 1, 1])),
      false,
    );
    let develop = RawDevelop { steps: vec![ProcessingStep::WhiteBalance], ..Default::default() };
    assert_eq!(three_colours(develop.develop_intermediate(&raw).unwrap()), [[0.0, 0.5, 0.75], [0.25, 0.0, 0.0]]);
  }
}
