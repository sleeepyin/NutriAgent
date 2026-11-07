# Human review pack: 37/50 code-swapped multi-component dishes

Automatic Exact / Near / Wrong labels are suggestions only. They do not close the audit on the 37/50 code-swapped multi-component dishes. A person has to confirm every review row.

Component rows that need a person: **68**. Dishes: **37**. The auto label column is a suggestion. `needs_human_review` is yes on every row below.

What a person confirms:

1. Automatic Exact / Near / Wrong labels are suggestions only. They do not close the audit on the 37/50 code-swapped multi-component dishes. A person has to confirm every review row.
2. Confirm which cafe ingredient, if any, the vision query was naming. The match uses food-noun overlap with the query. It has no synonym table, so steak against beef, yam against sweet potato, and salmon against fish stay unmatched or Wrong until a person links them.
3. Confirm Exact: the food nouns match, and the code does not add a preparation, cut, or style the cafe name lacks. Raw, fresh, NFS, NS, and plain are treated as unspecified.
4. Confirm Near: same food nouns with a different preparation, cut, or style; one name's food nouns contained in the other; or a variety/color difference such as sweet versus plain potato, white versus brown rice, or chinese broccoli. Near is not an identity success. A qualifier can name a different commodity.
5. Confirm Wrong: the query shares no food noun with any cafe ingredient, the code negates a cafe noun such as no cheese, the code is a meat analog the cafe ingredient is not, or the leftover nouns are not a subset or a variety qualifier.
6. A smaller calorie error is not evidence that the swapped code is the right food. Rows with energy_improved_but_identity_not_all_exact are the disagreement set and have to be read one by one.
7. Ambiguous ties, and matches under the existing 8 g / 8 kcal ingredient floor, need a person to pick the referent. Deprecated cafe rows are ignored.

Disagreement plates (calorie error fell, swapped codes are not all auto-Exact): **30**.

## Dishes

| dish | codes | E/N/W | worst | all exact | kcal AE change | disagreement | needs human |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dish_1557862783 | 2 | 0/1/1 | Wrong | no | -12.0 | yes | yes |
| dish_1558031394 | 1 | 0/1/0 | Near | no | -10.4 | yes | yes |
| dish_1558373159 | 1 | 0/1/0 | Near | no | -1.9 | yes | yes |
| dish_1558630325 | 2 | 0/2/0 | Near | no | -137.0 | yes | yes |
| dish_1558720236 | 3 | 0/3/0 | Near | no | -123.2 | yes | yes |
| dish_1558722247 | 2 | 0/2/0 | Near | no | -114.7 | yes | yes |
| dish_1559849466 | 1 | 0/1/0 | Near | no | -3.8 | yes | yes |
| dish_1559850338 | 2 | 0/2/0 | Near | no | -22.5 | yes | yes |
| dish_1560367980 | 3 | 0/3/0 | Near | no | -18.5 | yes | yes |
| dish_1560368464 | 1 | 0/0/1 | Wrong | no | -192.6 | yes | yes |
| dish_1561578800 | 1 | 0/0/1 | Wrong | no | -18.3 | yes | yes |
| dish_1561578865 | 3 | 0/3/0 | Near | no | -19.6 | yes | yes |
| dish_1561663580 | 1 | 0/1/0 | Near | no | -147.4 | yes | yes |
| dish_1562172600 | 2 | 0/1/1 | Wrong | no | -219.7 | yes | yes |
| dish_1563909580 | 2 | 0/2/0 | Near | no | -27.7 | yes | yes |
| dish_1563984167 | 1 | 0/0/1 | Wrong | no | -258.5 | yes | yes |
| dish_1563984199 | 1 | 0/0/1 | Wrong | no | -209.0 | yes | yes |
| dish_1563984242 | 2 | 0/1/1 | Wrong | no | -252.3 | yes | yes |
| dish_1563996485 | 1 | 0/1/0 | Near | no | -26.3 | yes | yes |
| dish_1565020822 | 2 | 0/1/1 | Wrong | no | -92.3 | yes | yes |
| dish_1565638550 | 1 | 0/1/0 | Near | no | -172.5 | yes | yes |
| dish_1565811000 | 2 | 0/2/0 | Near | no | -161.6 | yes | yes |
| dish_1566316726 | 1 | 0/1/0 | Near | no | -80.6 | yes | yes |
| dish_1566316757 | 2 | 0/1/1 | Wrong | no | -75.0 | yes | yes |
| dish_1566328745 | 1 | 0/1/0 | Near | no | -186.6 | yes | yes |
| dish_1566328776 | 2 | 0/2/0 | Near | no | -165.6 | yes | yes |
| dish_1566328805 | 2 | 0/2/0 | Near | no | -169.1 | yes | yes |
| dish_1566920339 | 1 | 0/0/1 | Wrong | no | -84.5 | yes | yes |
| dish_1568305257 | 3 | 0/0/3 | Wrong | no | -184.3 | yes | yes |
| dish_1568401167 | 1 | 0/0/1 | Wrong | no | -13.1 | yes | yes |
| dish_1560367926 | 1 | 0/1/0 | Near | no | +19.3 | no | yes |
| dish_1560368019 | 4 | 0/4/0 | Near | no | +97.0 | no | yes |
| dish_1560526514 | 3 | 0/2/1 | Wrong | no | +43.6 | no | yes |
| dish_1560801829 | 4 | 0/2/2 | Wrong | no | +22.5 | no | yes |
| dish_1562099134 | 2 | 0/2/0 | Near | no | +11.9 | no | yes |
| dish_1563389600 | 1 | 0/1/0 | Near | no | +8.0 | no | yes |
| dish_1565898158 | 3 | 0/2/1 | Wrong | no | +41.1 | no | yes |

## Swapped codes

| dish | query | swap | gt food code | gt name | auto | rationale | disagreement | needs human |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dish_1557862783 | beef | Beef, NFS → Beef, oxtails | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'beef' | yes | yes |
| dish_1557862783 | chicken | Chicken, back → Chicken, neck or ribs | ingr_0000000028 | chicken | Near | same food nouns; cut code {neck, rib} cafe {} | yes | yes |
| dish_1558031394 | roasted sweet potato | Sweet potato, NFS → Sweet potato, canned, fat added | ingr_0000000005 | potatoes | Near | shared {potato}; extra code {sweet}; extra cafe {} | yes | yes |
| dish_1558373159 | almonds | Almonds, NFS → Almonds, salted | ingr_0000000077 | almonds | Near | same food nouns; prep code-only {salted} cafe-only {} | yes | yes |
| dish_1558630325 | almonds | Almonds, NFS → Almonds, unroasted | ingr_0000000077 | almonds | Near | same food nouns; prep code-only {unroasted} cafe-only {} | yes | yes |
| dish_1558630325 | apple | Apple, raw → Apple, dried | ingr_0000000054 | apple | Near | same food nouns; prep code-only {dried} cafe-only {} | yes | yes |
| dish_1558720236 | carrot | Carrots, raw → Carrots, fresh, cooked with butter or margarine | ingr_0000000045 | carrot | Near | shared {carrot}; extra code {butter, margarine}; extra cafe {} | yes | yes |
| dish_1558720236 | cauliflower | Cauliflower, raw → Cauliflower, fresh, cooked with butter or margarine | ingr_0000000007 | cauliflower | Near | shared {cauliflower}; extra code {butter, margarine}; extra cafe {} | yes | yes |
| dish_1558720236 | apple | Apple, raw → Apple, baked | ingr_0000000054 | apple | Near | same food nouns; prep code-only {baked} cafe-only {} | yes | yes |
| dish_1558722247 | apple | Apple, raw → Apple, baked | ingr_0000000054 | apple | Near | same food nouns; prep code-only {baked} cafe-only {} | yes | yes |
| dish_1558722247 | cauliflower | Cauliflower, raw → Cauliflower, fresh, cooked with butter or margarine | ingr_0000000007 | cauliflower | Near | shared {cauliflower}; extra code {butter, margarine}; extra cafe {} | yes | yes |
| dish_1559849466 | almonds | Almonds, NFS → Almonds, unsalted | ingr_0000000077 | almonds | Near | same food nouns; prep code-only {unsalted} cafe-only {} | yes | yes |
| dish_1559850338 | broccoli | Broccoli, raw → Broccoli, chinese, raw | ingr_0000000027 | broccoli | Near | shared {broccoli}; extra code {chinese}; extra cafe {} | yes | yes |
| dish_1559850338 | almonds | Almonds, NFS → Almonds, unroasted | ingr_0000000077 | almonds | Near | same food nouns; prep code-only {unroasted} cafe-only {} | yes | yes |
| dish_1560367980 | spinach | Spinach, raw → Spinach, fresh, cooked, no added fat | ingr_0000000036 | spinach (raw) | Near | same food nouns; prep code-only {cooked} cafe-only {} | yes | yes |
| dish_1560367980 | asparagus | Asparagus, raw → Asparagus, fresh, cooked, fat added, NS as to fat type | ingr_0000000014 | asparagus | Near | same food nouns; prep code-only {cooked} cafe-only {} | yes | yes |
| dish_1560367980 | pizza | Pizza, cheese, stuffed crust → Pizza, no cheese, thin crust | ingr_0000000380 | pizza | Near | same food nouns; style code {crust, thin} cafe {}; code negates {cheese} | yes | yes |
| dish_1560368464 | chicken | Chicken, back → Chicken, canned, meat only | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'chicken' | yes | yes |
| dish_1561578800 | cheese pizza | Pizza, cheese, stuffed crust → Pizza, no cheese, thick crust | ingr_0000000011 | cheese pizza | Wrong | code negates {cheese}, which is in the cafe name | yes | yes |
| dish_1561578865 | penne pasta | Pasta, cooked → Pasta, vegetable, cooked | ingr_0000000074 | pasta | Near | shared {pasta}; extra code {vegetable}; extra cafe {} | yes | yes |
| dish_1561578865 | broccoli | Broccoli, raw → Broccoli, chinese, raw | ingr_0000000027 | broccoli | Near | shared {broccoli}; extra code {chinese}; extra cafe {} | yes | yes |
| dish_1561578865 | pizza | Pizza, cheese, stuffed crust → Pizza, extra cheese, thick crust | ingr_0000000011 | cheese pizza | Near | same food nouns; style code {crust, extra, thick} cafe {} | yes | yes |
| dish_1561663580 | chicken | Chicken, back → Chicken, canned, meat only | ingr_0000000028 | chicken | Near | same food nouns; prep code-only {canned} cafe-only {}; cut code {meat, only} cafe {} | yes | yes |
| dish_1562172600 | cottage cheese | Cottage cheese, farmer's → Cheese, cottage, dry curd | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'cottage cheese' | yes | yes |
| dish_1562172600 | roasted potatoes | Potato, roasted, NFS → Potato, roasted, from fresh, peel eaten, no added fat | ingr_0000000433 | roasted potatoes | Near | same food nouns; cut code {peel} cafe {} | yes | yes |
| dish_1563909580 | green beans | Green beans, raw → Green beans, fresh, cooked, fat added, NS as to fat type | ingr_0000000043 | green beans | Near | same food nouns; prep code-only {cooked} cafe-only {} | yes | yes |
| dish_1563909580 | spinach | Spinach, raw → Spinach, fresh, cooked, no added fat | ingr_0000000036 | spinach (raw) | Near | same food nouns; prep code-only {cooked} cafe-only {} | yes | yes |
| dish_1563984167 | cottage cheese | Cottage cheese, farmer's → Cheese, cottage, dry curd | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'cottage cheese' | yes | yes |
| dish_1563984199 | cottage cheese | Cottage cheese, farmer's → Cheese, cottage, dry curd | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'cottage cheese' | yes | yes |
| dish_1563984242 | cottage cheese | Cottage cheese, farmer's → Cheese, cottage, dry curd | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'cottage cheese' | yes | yes |
| dish_1563984242 | roasted potatoes | Potato, roasted, NFS → Potato, roasted, from fresh, peel eaten, no added fat | ingr_0000000437 | sweet potato | Near | shared {potato}; extra code {}; extra cafe {sweet} | yes | yes |
| dish_1563996485 | cauliflower | Cauliflower, raw → Cauliflower, fresh, cooked, fat added, NS as to fat type | ingr_0000000007 | cauliflower | Near | same food nouns; prep code-only {cooked} cafe-only {} | yes | yes |
| dish_1565020822 | bacon | Bacon, for use on a sandwich → Bacon, for use with vegetables | ingr_0000000004 | bacon | Near | same food nouns; style code {vegetable} cafe {} | yes | yes |
| dish_1565020822 | cottage cheese | Cottage cheese, farmer's → Cheese, cottage, dry curd | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'cottage cheese' | yes | yes |
| dish_1565638550 | chicken | Chicken, back → Chicken, canned, meat only | ingr_0000000244 | chicken breast | Near | same food nouns; prep code-only {canned} cafe-only {}; cut code {meat, only} cafe {breast} | yes | yes |
| dish_1565811000 | chicken | Chicken, back → Chicken, ground | ingr_0000000028 | chicken | Near | same food nouns; cut code {ground} cafe {} | yes | yes |
| dish_1565811000 | spinach | Spinach, raw → Spinach, fresh, cooked with butter or margarine | ingr_0000000036 | spinach (raw) | Near | shared {spinach}; extra code {butter, margarine}; extra cafe {} | yes | yes |
| dish_1566316726 | roasted potatoes | Potato, roasted, NFS → Potato, roasted, from fresh, peel eaten, no added fat | ingr_0000000437 | sweet potato | Near | shared {potato}; extra code {}; extra cafe {sweet} | yes | yes |
| dish_1566316757 | diced potatoes | Potato, NFS → Potato, baked, NFS | ingr_0000000437 | sweet potato | Near | shared {potato}; extra code {}; extra cafe {sweet} | yes | yes |
| dish_1566316757 | gravy | Gravy, NFS → Gravy, poultry | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'gravy' | yes | yes |
| dish_1566328745 | chicken | Chicken, back → Chicken, canned, meat only | ingr_0000000028 | chicken | Near | same food nouns; prep code-only {canned} cafe-only {}; cut code {meat, only} cafe {} | yes | yes |
| dish_1566328776 | pizza | Pizza, cheese, stuffed crust → Pizza, extra cheese, thick crust | ingr_0000000380 | pizza | Near | shared {pizza}; extra code {cheese}; extra cafe {} | yes | yes |
| dish_1566328776 | chicken | Chicken, back → Chicken, canned, meat only | ingr_0000000028 | chicken | Near | same food nouns; prep code-only {canned} cafe-only {}; cut code {meat, only} cafe {} | yes | yes |
| dish_1566328805 | pizza | Pizza, cheese, stuffed crust → Pizza, extra cheese, thick crust | ingr_0000000380 | pizza | Near | shared {pizza}; extra code {cheese}; extra cafe {} | yes | yes |
| dish_1566328805 | chicken | Chicken, back → Chicken, canned, meat only | ingr_0000000028 | chicken | Near | same food nouns; prep code-only {canned} cafe-only {}; cut code {meat, only} cafe {} | yes | yes |
| dish_1566920339 | cauliflower | Cauliflower, raw → Cauliflower, fresh, cooked with butter or margarine | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'cauliflower' | yes | yes |
| dish_1568305257 | sliced beef | Beef, sliced, with gravy, potatoes, vegetable, frozen meal → Beef, ground | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'sliced beef' | yes | yes |
| dish_1568305257 | sautéed spinach | Spinach, raw → Spinach, fresh, cooked, fat added, NS as to fat type | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'sautéed spinach' | yes | yes |
| dish_1568305257 | shredded carrots | Carrots, raw → Carrots, fresh, cooked with butter or margarine | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'shredded carrots' | yes | yes |
| dish_1568401167 | shredded carrots | Carrots, raw → Carrots, fresh, cooked with butter or margarine | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'shredded carrots' | yes | yes |
| dish_1560367926 | pizza | Pizza, cheese, stuffed crust → Pizza, no cheese, thick crust | ingr_0000000380 | pizza | Near | same food nouns; style code {crust, thick} cafe {}; code negates {cheese} | no | yes |
| dish_1560368019 | spinach | Spinach, raw → Spinach, fresh, cooked, fat added, NS as to fat type | ingr_0000000036 | spinach (raw) | Near | same food nouns; prep code-only {cooked} cafe-only {} | no | yes |
| dish_1560368019 | asparagus | Asparagus, raw → Asparagus, fresh, cooked, fat added, NS as to fat type | ingr_0000000014 | asparagus | Near | same food nouns; prep code-only {cooked} cafe-only {} | no | yes |
| dish_1560368019 | stuffed eggplant | Eggplant, raw → Eggplant, cooked, fat added | ingr_0000000335 | eggplant | Near | same food nouns; prep code-only {cooked} cafe-only {} | no | yes |
| dish_1560368019 | pizza | Pizza, cheese, stuffed crust → Pizza, extra cheese, thin crust | ingr_0000000380 | pizza | Near | shared {pizza}; extra code {cheese}; extra cafe {} | no | yes |
| dish_1560526514 | spinach | Spinach, raw → Spinach, fresh, cooked, fat added, NS as to fat type | ingr_0000000036 | spinach (raw) | Near | same food nouns; prep code-only {cooked} cafe-only {} | no | yes |
| dish_1560526514 | strawberry | Strawberries, raw → Strawberries, canned | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'strawberry' | no | yes |
| dish_1560526514 | pineapple | Pineapple, raw → Pineapple, canned, in syrup | ingr_0000000038 | pineapple | Near | shared {pineapple}; extra code {syrup}; extra cafe {} | no | yes |
| dish_1560801829 | potato | Potato, NFS → Potato, mashed, NFS | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'potato' | no | yes |
| dish_1560801829 | cauliflower | Cauliflower, raw → Cauliflower, fresh, cooked with butter or margarine | ingr_0000000007 | cauliflower | Near | shared {cauliflower}; extra code {butter, margarine}; extra cafe {} | no | yes |
| dish_1560801829 | pepper | Pepper, raw, NFS → Pepper, hot chili, raw | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'pepper' | no | yes |
| dish_1560801829 | pizza | Pizza, cheese, stuffed crust → Pizza, no cheese, thick crust | ingr_0000000392 | pepperoni pizza | Near | shared {pizza}; extra code {}; extra cafe {pepperoni} | no | yes |
| dish_1562099134 | carrot slices | Carrots, raw → Carrots, fresh, cooked with butter or margarine | ingr_0000000045 | carrot | Near | shared {carrot}; extra code {butter, margarine}; extra cafe {} | no | yes |
| dish_1562099134 | roasted chicken leg | Chicken leg, drumstick and thigh, stewed, skin eaten → Chicken leg, drumstick and thigh, sauteed, skin not eaten | ingr_0000000155 | chicken thighs | Near | same food nouns; prep code-only {sauteed} cafe-only {}; cut code {drumstick, leg, skin, thigh} cafe {thigh} | no | yes |
| dish_1563389600 | chicken | Chicken, back → Chicken, canned, meat only | ingr_0000000105 | grilled chicken | Near | same food nouns; prep code-only {canned} cafe-only {grilled}; cut code {meat, only} cafe {} | no | yes |
| dish_1565898158 | green beans | Green beans, raw → Green beans, NS as to form, cooked | ingr_0000000043 | green beans | Near | same food nouns; prep code-only {cooked} cafe-only {} | no | yes |
| dish_1565898158 | rice | Rice, cooked, NFS → Rice, white, cooked, glutinous | ingr_0000000023 | brown rice | Near | shared {rice}; qualifier code {glutinous, white} cafe {brown} | no | yes |
| dish_1565898158 | fish | Fish, raw → Fish, pike | (none) | (unmatched) | Wrong | no cafe ingredient shares a food noun with the query 'fish' | no | yes |
