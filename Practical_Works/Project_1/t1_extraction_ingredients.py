import re

ingredients_fn = "./data/ingredients.txt"
ingredients_solutions_fn = "./data/ingredients_solutions.txt"

pattern_0 = "^(Préparation|Finition) (.+)"
sub_pattern_0 = ""
pattern_1 = "^((?:\d+(?:\/\d+)?|\d+(?:,\d+))?\s(?:oz|feuilles|feuille)?\s?à\s(?:\d+(?:\/\d+)?|\d+(?:,\d+))\s(?:oz|feuilles|feuille)?)\s?(?:de\s|d[’'])?(.+)"
sub_pattern_1 = ""
pattern_2 = "^((?:\d+(?:,\d+)?\s(?:m[lL]|g|kg|(?:(?:cuillères|cuillère|c\.) à (?:café|thé|soupe|c\.|s\.))?|tasses|tasse|boîtes de conserve))?\s?\(?(?:\d+(?:\/\d+)?|\d+(?:,\d+)?|(?:\d+)?\s?.)?\s(?:(?:(?:cuillères|cuillère|c.) à (?:café|thé|soupe|c.|s.|\.s))|(?:g|m[lL]|oz|lb|tasses|tasse|cl))\)?)\s(?:d[’']|de)?\s?(.+)"
sub_pattern_2 = ""
pattern_3 = "^((?:\d+(?:\/\d+)?|\d+(?:,\d+)?|(?:\d+)?\s?.)?\s(?:enveloppe|bottes|botte|pièces|pièce|pincées|pincée|Bouquet|Rondelle|gousses|gousse|tranches|verres))\s(?:de|d[’'])?\s?(.+)"
sub_pattern_3 = ""
pattern_4 = "^([^0-9].+)\s(?:de\s|d[’'])((?:un|une|deux|trois)\s(?:demi|tiers|quart))-?(.+)"
sub_pattern_4 = ""
pattern_5 = "^(\d+(?:\/\d+)?|\d+(?:,\d+))?\s(.+)"
sub_pattern_5 = ""
pattern_6 = "^((?:\d+(?:\/\d+)?|\d+(?:,\d+)?|(?:\d+)?\s?.)?\s(?:tranches|tronçons))\s(?:de|d[’'])?\s?(.+)\s(\d+ g)"
sub_pattern_6 = ""
pattern_7 = "(.+(?:(?<=Feuilles)|(?<=Feuille)))\s(?:de|d[’'])\s?(.+)"
sub_pattern_7 = ""
pattern_8 = "(.+(?:(?<=Quelques)|(?<=Quelque)))\s?(.+)"
sub_pattern_8 = ""
pattern_11 = "(^[^0-9].+)"
sub_pattern_11 = ""

pattern_a = "(.+)\s(ou au goût)"
pattern_b = "((?:.+)?(?:bananes|banane|pommes vertes|pommes rouges|pommes jaunes|échalotes|échalote|champignons|champignon|poulet|oeufs|oeuf|gingembre|oignon vert|eau bouillante|pommes de terre|estragon frais))\s?.+(en\s.+)"
pattern_c = "((?:.+)?(?:champignons|champignon|échalotes|échalote|oeufs|oeuf|gingembre|fromage cheddar|poivre noir|cacahuètes|palourdes|langoustines surgelées|asperges moyennes|persil plat|pain baguette|Beurre|amandes))\s(.+)"
pattern_d = "((?:.+)?(?:oignons perlés|asperges|oignons verts|petit oignon rouge|ail|moules|tomates cerises jaunes)),\s(.+)"
pattern_e = "(.+)\s((?<!ou\s)au goût)"
pattern_f = "(.+),.+\s(\(facultatif\))"
pattern_g = "(.+)\s(?:de|d[’'])\s?(\d kg)"

all_regex = [("type_z", pattern_0, sub_pattern_0),
             ("type_a", pattern_1, sub_pattern_1),
             ("type_a", pattern_2, sub_pattern_2),
             ("type_d", pattern_6, sub_pattern_6),
             ("type_a", pattern_3, sub_pattern_3),
             ("type_a", pattern_4, sub_pattern_4),
             ("type_a", pattern_5, sub_pattern_5),
             ("type_e", pattern_7, sub_pattern_7),
             ("type_a", pattern_8, sub_pattern_8),
             ("no_quantity", pattern_11, sub_pattern_11)]

ingr_regex = [("ou", pattern_a), ("en", pattern_b), ("none", pattern_c), (",", pattern_d), ("ou", pattern_e), ("ou", pattern_f), ("()", pattern_g)]

def load_ingredients(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_items = f.readlines()
    ingredients = [x.strip() for x in raw_items]
    return ingredients

def get_ingredients(text):
    for tag, regex, substitution in all_regex:
        x = re.match(regex, text)
        if x:
            more_qty = ""
            clean_ingr = ""
            if tag == "no_quantity":
                qty = ""
                more_qty, clean_ingr = clean_ingredient(x.group(1))
            else:
                qty, first_pass = getSubs(tag, x)
                more_qty, clean_ingr = clean_ingredient(first_pass)
            final_qty = ""
            if (more_qty.strip() != "") and (qty != ""):
                final_qty = qty + " " + more_qty
            elif more_qty != "":
                final_qty = more_qty
            else:
                final_qty = qty
            return final_qty, clean_ingr
    return "", ""

def getSubs(tag, x):
    if tag == "type_b":
        quantity = x.group(1).strip() + x.group(3)
        ingredient = x.group(2)
        return quantity, ingredient
    elif tag == "type_c":
        quantity = x.group(1).strip() + x.group(2).strip()
        ingredient = x.group(3)
        return quantity, ingredient
    elif tag == "type_d":
        quantity = x.group(1).strip() + " (" + x.group(3).strip() + ")"
        ingredient = x.group(2)
        return quantity, ingredient
    elif tag == "type_e":
        return x.group(1), x.group(2)
    elif tag == "type_z":
        return "", ""

    ingredient = x.group(2)
    quantity = x.group(1).strip()
    return quantity, ingredient

def clean_ingredient(text):
    for tag, regex in ingr_regex:
        y = re.match(regex, text)
        if y:
            if tag == "one":
                return "", y.group(1)
            else:
                return getIngredientOnly(tag, y)
    return "", text

def getIngredientOnly(tag, x):
    if tag == "ou":
        return x.group(2).strip(), x.group(1)
    elif tag == "()":
        return "(" + x.group(2).strip() + ")", x.group(1)
    elif tag == "en":
        return "", x.group(1)
    elif tag == "de":
        ingredient = x.group(2)
        return "", ingredient
    elif tag == ",":
        ingredient = x.group(1)
        return "", ingredient
    elif tag == "none":
        ingredient = x.group(1)
        return "", ingredient

    return "", x

if __name__ == '__main__':
    recipe = load_ingredients(ingredients_fn)
    solutions = load_ingredients(ingredients_solutions_fn)
    quantity_answer = [solution.split('   ')[1].split(':')[-1] for solution in solutions]
    ingredients_answer = [solution.split('   ')[2].split(':')[-1] for solution in solutions]
    quantity_score = 0
    ingredients_score = 0

    for i, instruction in enumerate(recipe):
        if instruction:
            quantity, ingredient = get_ingredients(instruction)
            quantity_score += 1 if quantity == quantity_answer[i] else 0
            ingredients_score += 1 if ingredient == ingredients_answer[i] else 0
            #print("\t{}\t QUANTITE: {}\t INGREDIENT: {}".format(instruction, quantity, ingredient))
    quantity_score /= len(solutions)
    ingredients_score /= len(solutions)
    print("Score en quantité: {0} | Score en ingrédients: {1}".format(quantity_score, ingredients_score))
