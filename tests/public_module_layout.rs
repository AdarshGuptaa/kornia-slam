use kornia_slam::frame::OrbFeatures;
use kornia_slam::mapping::{Keyframe, Map, MapPoint};
use kornia_slam::odometry::bootstrap::{
    BootstrapConfig, BootstrapOutcome, BootstrapRejectReason, try_bootstrap,
};
use kornia_slam::odometry::estimation::MapProjectionEstimator;
use kornia_slam::odometry::estimation::map_projection::{KeyframePolicy, OdometryConfig};
use kornia_slam::odometry::estimation::pnp::PnpConfig;
use kornia_slam::{OdometryResult, OdometryStatus};

#[test]
fn exposes_restructured_public_modules() {
    fn assert_public_api(
        _: Option<OdometryResult>,
        _: Option<OdometryStatus>,
        _: Option<OrbFeatures>,
        _: Option<BootstrapConfig>,
        _: Option<BootstrapOutcome>,
        _: Option<BootstrapRejectReason>,
        _: Option<PnpConfig>,
        _: Option<KeyframePolicy>,
        _: Option<OdometryConfig>,
        _: Option<Map>,
        _: Option<Keyframe>,
        _: Option<MapPoint>,
        _: Option<MapProjectionEstimator>,
    ) {
    }

    let _ = try_bootstrap;

    assert_public_api(
        None, None, None, None, None, None, None, None, None, None, None, None, None,
    );
}
