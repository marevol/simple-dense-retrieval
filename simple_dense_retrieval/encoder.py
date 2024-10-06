from collections import Counter


class LabelEncoder:
    def __init__(self, min_frequency=1):
        """
        Custom ordinal encoder for categorical values with frequency-based filtering.

        Args:
            min_frequency (int): The minimum frequency for a category to be encoded.
        """
        self.min_frequency = min_frequency
        self.category_to_index = {}
        self.is_fitted = False

    def fit(self, categories):
        """
        Fit the encoder on a list of categories, considering only categories with a frequency
        greater than or equal to min_frequency.

        Args:
            categories (list or array-like): The list of categories to fit on.
        """
        # Count the frequency of each category
        category_counts = Counter(categories)

        # Manually manage the index, starting from 1
        idx = 1
        for category, count in category_counts.items():
            if count >= self.min_frequency and category is not None and category != "":
                self.category_to_index[category] = idx
                idx += 1

        self.is_fitted = True

    def transform(self, categories):
        """
        Transform categories to integer values.

        Args:
            categories (list or array-like): The list of categories to transform.

        Returns:
            list: The list of integer values corresponding to the categories.
        """
        if not self.is_fitted:
            raise RuntimeError("The encoder must be fitted before calling transform().")

        return [self.category_to_index.get(category, 0) for category in categories]

    def __len__(self):
        """
        Return the number of unique categories that were encoded.
        """
        return len(self.category_to_index) + 1
