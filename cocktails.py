from typing import AbstractSet, Set, FrozenSet, Optional

Cocktail = FrozenSet[str]
Cocktails = AbstractSet[Cocktail]
Ingredients = Set[str]


class BranchBound(object):
    def __init__(self, max_calls: int, max_size: int) -> None:
        self.calls: int = max_calls
        self.max_size: int = max_size

        self.highest_score: int = 0
        self.highest: Cocktails = set()

        self.min_amortized_cost: dict[Cocktail, float] = {}
        self.min_cover: dict[Cocktail, int] = {}

        self.counters_w = {"a": 0, "b": 0, "c": 0, None: 0}
        self.counters = {"a": 0, "b": 0, "c": 0, None: 0}

    def search(
        self,
        candidates: Cocktails,
        partial: Optional[Cocktails] = None,
        forbidden: Optional[Cocktails] = None,
    ) -> Cocktails:

        if partial is None:
            # We'll only hit this conditional on the first call, so
            # we set up our cocktail scoring dictionary
            partial = set()
            forbidden = set()
            cardinality: dict[str, int] = {}
            for cocktail in candidates:
                for ingredient in cocktail:
                    if ingredient in cardinality:
                        cardinality[ingredient] += 1
                    else:
                        cardinality[ingredient] = 1

            # For each cocktail, we can calculate the minimum
            # amoritized cost. That is, if we were to have enough
            # ingredients to make all the cocktails, how much should
            # we pay, in ingredient-cost, for each cocktail. For
            # example, if a cocktail has a unique ingredient, and two
            # other ingredients shared by one other cocktail, then the
            # amortized cost would be 1/1 + 1/2 + 1/2 = 2
            #
            # The minimum amoritized cost is a lower bound to how much
            # we will ever pay in ingredient cost for a cocktail.
            for cocktail in candidates:
                self.min_amortized_cost[cocktail] = sum(
                    1 / cardinality[ingredient] for ingredient in cocktail
                )
                self.min_cover[cocktail] = min(
                    cardinality[ingredient] for ingredient in cocktail
                )

        if self.calls <= 0:
            print("early stop")
            return self.highest

        self.calls -= 1

        score = len(partial)

        if score > self.highest_score:

            self.highest = partial
            self.highest_score = score
            print(sorted(cocktails[k] for k in self.highest))
            print(self.highest_score)

        # what cocktails could be added without going over our
        # ingredient budget?
        assert forbidden is not None

        partial_ingredients: Ingredients
        partial_ingredients = set().union(*partial)
        candidates = {
            cocktail
            for cocktail in candidates
            if len(cocktail | partial_ingredients) <= self.max_size
        }
        filtered_candidates = set()
        for cocktail in candidates:
            extended_ingredients = cocktail | partial_ingredients
            if len(extended_ingredients) <= self.max_size:
                # when we branch, we need to not only remove
                # a cocktail from the candidate set, but make
                # it impossible that final ingredient list could
                # be a superset of the cocktail. failing this, we
                # could undercount the score of branch.
                #
                # unfortunately, this is an O(N^2) operation and
                # make things pretty slow
                forbidden_cover = any(
                    forbidden_cocktail <= extended_ingredients
                    for forbidden_cocktail in forbidden
                )
                if not forbidden_cover:
                    filtered_candidates.add(cocktail)

        candidates = filtered_candidates

        keep_exploring = self.keep_exploring(candidates, partial, partial_ingredients)

        if candidates and keep_exploring:

            # the best heuristic i've found is to pick the candidates
            # with the smallest amortized cost
            best = min(candidates, key=lambda x: self.min_amortized_cost[x])

            new_partial_ingredients = partial_ingredients | best
            covered_candidates = {
                cocktail
                for cocktail in candidates
                if cocktail <= new_partial_ingredients
            }

            self.search(
                candidates - covered_candidates, partial | covered_candidates, forbidden
            )

            remaining = candidates - set([best])
            forbidden = forbidden | set([best])

            self.search(remaining, partial, forbidden)

        return self.highest

    def keep_exploring(
        self,
        candidates: Cocktails,
        partial: Cocktails,
        partial_ingredients: Ingredients,
    ) -> bool:

        threshold = self.highest_score - len(partial)

        bound_functions = (
            self.total_bound,
            self.concentration_bound,
            self.singleton_bound,
            self.amortization_bound,
        )
        for func in bound_functions:
            bound = func(candidates, partial, partial_ingredients)
            if bound <= threshold:
                return False

        return True

    def concentration_bound(
        self,
        candidates: Cocktails,
        partial: Cocktails,
        partial_ingredients: Ingredients,
    ) -> int:
        """
        best case is that excess ingredients are concentrated in
        some cocktails. if we are in this best case, then if we
        remove the cocktails that add the most new ingredients
        we'll be back under the ingredient budget

        note that we are just updating the bound. it could actually
        be that we want to add one of these cocktails that add
        a lot of ingredients
        """

        candidate_ingredients: Ingredients
        candidate_ingredients = set().union(*candidates)
        excess_ingredients = (
            len(candidate_ingredients | partial_ingredients) - self.max_size
        )

        ingredient_increases = sorted(
            (len(cocktail - partial_ingredients) for cocktail in candidates),
            reverse=True,
        )

        upper_increment = len(candidates)
        for ingredient_increase in ingredient_increases:
            if excess_ingredients <= 0:
                break
            upper_increment -= 1
            excess_ingredients -= ingredient_increase

        return upper_increment

    def total_bound(
        self,
        candidates: Cocktails,
        partial: Cocktails,
        partial_ingredients: Ingredients,
    ) -> int:
        return len(candidates)

    def singleton_bound(
        self,
        candidates: Cocktails,
        partial: Cocktails,
        partial_ingredients: Ingredients,
    ) -> int:
        """
        There are many cocktails that have an unique ingredient.

        Each cocktail with a unique ingredient will cost at least
        one ingredient from our ingredient budget, and the total
        possible increase due to these unique cocktails is bounded
        by the ingredient budget
        """

        n_unique_cocktails = sum(
            1 for cocktail in candidates if self.min_cover[cocktail] == 1
        )
        ingredient_budget = self.max_size - len(partial_ingredients)

        upper_increment = (
            len(candidates)
            - n_unique_cocktails
            + min(n_unique_cocktails, ingredient_budget)
        )

        return upper_increment

    def amortization_bound(
        self,
        candidates: Cocktails,
        partial: Cocktails,
        partial_ingredients: Ingredients,
    ) -> float:
        """
        we can see how many more cocktails we can fit into the ingredient
        budget assuming minimum, amortized costs for all cocktails
        """
        amortized_budget = self.max_size - sum(
            self.min_amortized_cost[cocktail] for cocktail in partial
        )
        candidate_amortized_costs = sorted(
            self.min_amortized_cost[cocktail] for cocktail in candidates
        )

        total_cost = 0.0
        upper_increment = 0
        for cost in candidate_amortized_costs:
            total_cost += cost
            if total_cost > amortized_budget:
                break
            upper_increment += 1

        return upper_increment


if __name__ == "__main__":
    import csv

    cocktails = {}

    with open("ingredients.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            name, *ingredients = row
            cocktails[frozenset(ingredients)] = name

    bb = BranchBound(8000000, 30)
    best = bb.search(cocktails.keys())

    print(sorted(set().union(*best)))
    print(sorted(cocktails[k] for k in best))
