"""Default Rerun viewer layout for the house-builder harness."""
import rerun.blueprint as rrb


def build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/harness/cameras/cam0", name="Cam0 (policy view)"),
                rrb.Spatial2DView(origin="/harness/cameras/cam1", name="Cam1 (verification view)"),
            ),
            rrb.Vertical(
                rrb.TextLogView(
                    origin="/harness",
                    name="State / Instructions / Verification",
                    contents=["/harness/state", "/harness/instruction", "/harness/verification"],
                ),
                rrb.TimeSeriesView(origin="/harness/verification", name="Verification Checks"),
                rrb.TimeSeriesView(origin="/harness/arm", name="Arm Joints"),
            ),
        ),
        collapse_panels=True,
    )
