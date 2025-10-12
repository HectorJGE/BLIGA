from dataclasses import dataclass
from typing import Dict, Set, List, Any, Optional

UserId = str
ItemId = str
Category = str

@dataclass
class RecsData:
    ratings: Dict[UserId, Dict[ItemId, float]]
    item_categories: Dict[ItemId, Set[Category]]
    # === NUEVO ===
    item_meta: Optional[Dict[ItemId, Dict[str, Any]]] = None

    def items(self) -> Set[ItemId]:
        rated_items = {i for ur in self.ratings.values() for i in ur.keys()}
        return set(self.item_categories.keys()) | rated_items | set(self.item_meta or {})

    def unrated_items_for(self, user: UserId) -> List[ItemId]:
        rated = set(self.ratings.get(user, {}).keys())
        return [i for i in self.items() if i not in rated]
