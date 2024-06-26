use crate::integrations::VectorLoaderError;
use crate::{VectorFile, VelloAsset};
use bevy::prelude::*;
use std::sync::Arc;

/// Deserialize a Lottie file from bytes.
pub fn load_lottie_from_bytes(bytes: &[u8]) -> Result<VelloAsset, VectorLoaderError> {
    // Load Lottie JSON bytes with the Velato (bodymovin) parser
    let composition = velato::Composition::from_slice(bytes).map_err(VectorLoaderError::Velato)?;

    let width = composition.width as f32;
    let height = composition.height as f32;

    let vello_vector = VelloAsset {
        file: VectorFile::Lottie(Arc::new(composition)),
        local_transform_center: {
            let mut transform = Transform::default();
            transform.translation.x = width / 2.0;
            transform.translation.y = -height / 2.0;
            transform
        },
        width,
        height,
        alpha: 1.0,
    };

    Ok(vello_vector)
}

/// Deserialize a Lottie file from a string slice.
pub fn load_lottie_from_str(json_str: &str) -> Result<VelloAsset, VectorLoaderError> {
    let bytes = json_str.as_bytes();

    load_lottie_from_bytes(bytes)
}
