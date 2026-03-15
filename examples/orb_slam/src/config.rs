use kornia_slam::estimation::map_projection::MapProjectionConfig;
use kornia_slam::estimation::two_view::TwoViewInitConfig;

/// Example-local pipeline preset used by the standalone ORB-SLAM binary.
pub struct PipelineConfig {
    pub two_view_init: TwoViewInitConfig,
    pub map_projection: MapProjectionConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        let mut two_view_init = TwoViewInitConfig::default();
        two_view_init
            .estimation_config
            .triangulation
            .max_midpoint_gap = 0.25;
        two_view_init
            .estimation_config
            .triangulation
            .max_reprojection_error = 3.0;

        Self {
            two_view_init,
            map_projection: MapProjectionConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PipelineConfig;

    #[test]
    fn default_pipeline_config_applies_example_overrides() {
        let config = PipelineConfig::default();

        assert_eq!(
            config
                .two_view_init
                .estimation_config
                .triangulation
                .max_midpoint_gap,
            0.25
        );
        assert_eq!(
            config
                .two_view_init
                .estimation_config
                .triangulation
                .max_reprojection_error,
            3.0
        );
        assert!(config.map_projection.enable_local_ba);
    }
}
