from .janus_pro_nodes import JanusProModelLoaderNode
from .janus_pro_nodes import JanusProMultimodalUnderstandingNode
from .janus_pro_nodes import JanusProImageGenerationNode

NODE_CLASS_MAPPINGS = {
    "JanusProModelLoaderNode": JanusProModelLoaderNode,
    "JanusProMultimodalUnderstandingNode": JanusProMultimodalUnderstandingNode,
    "JanusProImageGenerationNode": JanusProImageGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JanusProModelLoaderNode": "Janus Pro Model Loader",
    "JanusProMultimodalUnderstandingNode": "Janus Pro Multimodal Understanding",
    "JanusProImageGenerationNode": "Janus Pro Image Generation",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
