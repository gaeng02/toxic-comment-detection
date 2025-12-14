from typing import List 
import numpy as np  

from sentence_transformers import SentenceTransformer

class Embedder : 
    
    def __init__ (self, model_name : str = "jhgan/ko-sroberta-multitask", device : str = None) : 
        
        self.model = SentenceTransformer(model_name, device = device)

        
    def encode (self, texts : List[str]) -> np.ndarray : 
        if isinstance(texts, str) : 
            texts = [texts]
            
        return self.model.encode(texts, convert_to_numpy = True, show_progress_bar = False)
    

def l2_normalize (vectors : np.ndarray, eps : float = 1e-12) -> np.ndarray : 
    norms = np.linalg.norm(vectors, axis = 1, keepdims = True)
    norms = np.maximum(norms, eps)
    
    return vectors / norms
