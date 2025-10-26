from dataclasses import dataclass
from typing import Dict, Set, List, Any, Optional

UserId = str
ItemId = str
Category = str

@dataclass
class RecsData:
    ratings: Dict[UserId, Dict[ItemId, float]]
    item_categories: Dict[ItemId, Set[Category]]
    # metadatos crudos (títulos, cast, etc.)
    item_meta: Optional[Dict[ItemId, Dict[str, Any]]] = None
    # inyectados (opcionales)
    content_model: Any = None           # objeto con .item_vector(item_id) -> np.ndarray|None
    user_means: Optional[Dict[UserId, float]] = None  # si usás vector np.ndarray, podés guardar el mapping aparte

    def items(self) -> Set[ItemId]:
        rated_items = {i for ur in self.ratings.values() for i in ur.keys()}
        return set(self.item_categories.keys()) | rated_items | set((self.item_meta or {}).keys())

    def unrated_items_for(self, user: UserId) -> List[ItemId]:
        rated = set(self.ratings.get(user, {}).keys())
        return [i for i in self.items() if i not in rated]

    # helpers
    def item_vector(self, item: ItemId):
        return self.content_model.item_vector(item) if self.content_model else None

    def mean_of(self, user: UserId, default: float = 0.0) -> float:
        if isinstance(self.user_means, dict):
            return self.user_means.get(user, default)
        return default
