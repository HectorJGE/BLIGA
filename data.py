from dataclasses import dataclass
from typing import Dict, Set, List

UserId = str
ItemId = str
Category = str

@dataclass
class RecsData:
    """
    Representa las dos matrices base:
    - U_I: ratings[u][i] = nota (1..5)
    - I_C: item_categories[i] = {categorías...}
    """
    ratings: Dict[UserId, Dict[ItemId, float]]
    item_categories: Dict[ItemId, Set[Category]]

    def items(self) -> Set[ItemId]:
        rated_items = {i for ur in self.ratings.values() for i in ur.keys()}
        return set(self.item_categories.keys()) | rated_items

    def unrated_items_for(self, user: UserId) -> List[ItemId]:
        rated = set(self.ratings.get(user, {}).keys())
        return [i for i in self.items() if i not in rated]