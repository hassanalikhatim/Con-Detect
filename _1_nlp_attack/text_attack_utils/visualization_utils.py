def wordsChanged(result, color_method=None):
    """Highlights the difference between two texts using color.
    Has to account for deletions and insertions from original text to
    perturbed. Relies on the index map stored in
    ``result.original_result.attacked_text.attack_attrs["original_index_map"]``.
    """
    t1 = result.original_result.attacked_text
    t2 = result.perturbed_result.attacked_text
    
    if color_method is None:
        return t1.printable_text(), t2.printable_text()
    
    color_1 = result.original_result.get_text_color_input()
    color_2 = result.perturbed_result.get_text_color_perturbed()

    # iterate through and count equal/unequal words
    words_1_idxs = []
    t2_equal_idxs = set()
    original_index_map = t2.attack_attrs["original_index_map"]
    for t1_idx, t2_idx in enumerate(original_index_map):
        if t2_idx == -1:
            # add words in t1 that are not in t2
            words_1_idxs.append(t1_idx)
        else:
            w1 = t1.words[t1_idx]
            w2 = t2.words[t2_idx]
            if w1 == w2:
                t2_equal_idxs.add(t2_idx)
            else:
                words_1_idxs.append(t1_idx)

    # words to color in t2 are all the words that didn't have an equal,
    # mapped word in t1
    words_2_idxs = list(sorted(set(range(t2.num_words)) - t2_equal_idxs))

    # make lists of colored words
    words_1 = [t1.words[i] for i in words_1_idxs]
    original = result.original_result.attacked_text.printable_text().split()
    print("\t total words: ", len(original), end="")
    if len(original) == 0:
      return len(words_1), len(words_1)
    else:
      return len(words_1), len(words_1)/len(original)