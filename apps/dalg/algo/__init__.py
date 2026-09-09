from dalg.algo.controls import ConstantAlgorithm, ExactRangeAlgorithm
from dalg.algo.sgbm import SGBMAlgorithm, SGBMConfig
from dalg.algo.plane_sweep import PlaneSweepAlgorithm, PlaneSweepConfig
from dalg.algo.features import FeatureConfig, FeatureTriangulationAlgorithm
from dalg.algo.optical_flow import OpticalFlowConfig, OpticalFlowTriangulationAlgorithm
from dalg.algo.ground_plane import GroundPlaneAlgorithm, GroundPlaneConfig
from dalg.algo.monocular_depth import MonocularDepthAlgorithm, MonocularDepthConfig

ALGORITHMS = {"sgbm": SGBMAlgorithm, "constant": ConstantAlgorithm,
              "exact_range": ExactRangeAlgorithm,
              "plane_sweep": PlaneSweepAlgorithm,
              "feature_triangulation": FeatureTriangulationAlgorithm,
              "optical_flow_triangulation": OpticalFlowTriangulationAlgorithm,
              "ground_plane": GroundPlaneAlgorithm,
              "monocular_depth": MonocularDepthAlgorithm}
CONFIGS = {"sgbm": SGBMConfig, "plane_sweep": PlaneSweepConfig,
           "feature_triangulation": FeatureConfig,
           "optical_flow_triangulation": OpticalFlowConfig,
           "ground_plane": GroundPlaneConfig,
           "monocular_depth": MonocularDepthConfig}
