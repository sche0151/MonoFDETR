from .depth_predictor import DepthPredictor
from .marigold_depth_predictor import MarigoldDepthPredictor
from .dpt_depth_predictor import DPTDepthPredictor

from .ddn_loss import DDNLoss


def build_depth_predictor(config, position_embedding = None):
    depth_predictor_type = getattr(config, 'depth_predictor_type', 'Default')
    
    if depth_predictor_type == 'Default':
        depth_predictor = DepthPredictor(config)
    elif 'marigold' in depth_predictor_type:
        depth_predictor = MarigoldDepthPredictor(config, position_embedding)
    else:
        raise ValueError(f"Not supported {depth_predictor_type}")
    
    return depth_predictor
