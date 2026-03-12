use kornia_slam::frame::OrbFeatures;
use kornia_slam::map::{Keyframe, Map, MapPoint};
use kornia_slam::estimation::two_view::{
    TwoViewAcceptanceConfig, TwoViewInitConfig, TwoViewInitOutcome, TwoViewInitRejectReason,
    try_initialize_two_view,
};
use kornia_slam::estimation::MapProjectionEstimator;
use kornia_slam::estimation::map_projection::{
    KeyframePolicy, MapProjectionConfig, MapProjectionEstimateOutcome, MapProjectionRejectReason,
    PnpConfig,
};
use kornia_slam::{OdometryResult, OdometryStatus};

#[test]
fn exposes_restructured_public_modules() {
    fn assert_public_api(
        _: Option<OdometryResult>,
        _: Option<OdometryStatus>,
        _: Option<OrbFeatures>,
        _: Option<TwoViewAcceptanceConfig>,
        _: Option<TwoViewInitConfig>,
        _: Option<TwoViewInitOutcome>,
        _: Option<TwoViewInitRejectReason>,
        _: Option<PnpConfig>,
        _: Option<KeyframePolicy>,
        _: Option<MapProjectionConfig>,
        _: Option<MapProjectionEstimateOutcome>,
        _: Option<MapProjectionRejectReason>,
        _: Option<Map>,
        _: Option<Keyframe>,
        _: Option<MapPoint>,
        _: Option<MapProjectionEstimator>,
    ) {
    }

    let _ = try_initialize_two_view;

    assert_public_api(
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None,
    );
}
