You are taking over the ML/model-training part of our Amazon ML Challenge 2026 Business Entity Resolution project.

IMPORTANT:
Do not blindly rewrite or replace existing code. First inspect the repository and understand the current implementation. Preserve the existing preprocessing and blocking work unless there is a concrete technical reason to modify it.

==================================================
1. PROJECT
==================================================

Project:
Amazon ML Challenge 2026 – Business Entity Resolution

Goal:
For every Source 1 entity, identify all matching entities from Source 2 and Source 3.

The task allows:
- zero matches
- one match
- multiple matches

This is NOT a simple one-to-one entity matching problem.

The final system must produce:
1. matching_results.tsv
2. candidate_pairs.tsv

The final matching output must contain exactly one row for every Source 1 test entity.

==================================================
2. DATA
==================================================

Training files:

data/train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv

Test files:

data/test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv

All TSV files should be read using:

sep="\t"

Columns:

Source 1/2/3:
    entity_id
    business_name
    business_address
    country

Ground truth:
    source1_entity_id
    matched_entity_ids

IMPORTANT:
Do NOT assume country is only US and India. The test set also contains France, so country handling must remain open-set.

==================================================
3. DATASET SCALE
==================================================

Approximate training sizes:

Source 1:
2,206,821 rows

Source 2:
5,034,616 rows

Source 3:
5,285,603 rows

Ground truth:
2,206,821 rows

Therefore exhaustive comparison is impossible.

Do NOT create:

Source1 × Source2
or
Source1 × Source3

pairwise matrices.

Candidate generation/blocking is necessary.

==================================================
4. CURRENT GIT BRANCH
==================================================

The current work is already pushed to:

branch:
preprocessing

The branch is available on origin.

Pull it using:

git fetch origin
git checkout preprocessing
git pull origin preprocessing

==================================================
5. WORK ALREADY COMPLETED
==================================================

The preprocessing work has already been implemented.

Files:

src/preprocessing.py

src/blocking.py

notebooks/01_eda.ipynb

notebooks/02_blocking.ipynb

utils/validate_submission.py

requirements.txt

.gitignore

==================================================
6. PREPROCESSING IMPLEMENTATION
==================================================

src/preprocessing.py currently performs:

- missing-value handling
- Unicode normalization using NFKC
- lowercase conversion
- punctuation replacement
- whitespace normalization

It creates:

name_norm
address_norm
country_norm

It also creates:

name_compact
address_compact

The original columns are NOT overwritten.

IMPORTANT:
Do not ASCII-strip Unicode data.

The dataset can contain:
- accented characters
- Hindi/other Unicode text
- transliterated names

Preserve Unicode.

==================================================
7. CURRENT BLOCKING IMPLEMENTATION
==================================================

src/blocking.py currently contains an initial blocking implementation.

Current blocking signals include:

1. country + normalized name

2. country + compact name

3. country + name prefix

4. country + house number

The current implementation is only an INITIAL blocking implementation.

It has been tested on a real Source 1 entity:

S1-965667

For that entity the current basic blocking generated approximately:

S2 candidates: 1745
S3 candidates: 1916

This proves the blocking logic is functioning, but we have NOT yet completed full candidate recall evaluation.

IMPORTANT:
Do not assume this blocking implementation is already optimal.

Your first task should be to evaluate and improve candidate generation.

==================================================
8. IMPORTANT MATCHING EXAMPLE
==================================================

A known training example:

Source 1:

S1-965667

Business name:
Maure Williams Colombier Inc

Address:
85 Wayne Avenue, Ticonderoga, NY

Country:
US

Ground truth matches include:

S2-681193310
S2-743505751
S3-775321672
S3-11291185
S3-860443364

The corresponding records demonstrate important noise patterns:

- typo in business name
- missing address
- address typo
- business-name variation
- domain-style business name
- trade-name variation
- a candidate where address is much stronger than name

Therefore:

DO NOT require both name and address to match.

A true match may have:
- strong name + missing address
- weak name + strong address
- strong address + unusual business name
- typo-heavy name
- domain-like name

==================================================
9. YOUR FIRST TASK — CANDIDATE RECALL
==================================================

Before training the model, evaluate whether blocking is capable of finding the true matches.

For every sampled Source 1 entity:

1. Generate candidates from Source 2 and Source 3.

2. Compare generated candidate IDs against train_ground_truth.tsv.

3. Calculate candidate recall.

Candidate recall:

number of true ground-truth matches appearing in candidates
-------------------------------------------------------------
total number of ground-truth matches

This is critical.

If a true match is excluded during blocking, the ML model can NEVER recover it later.

Therefore candidate generation should prioritize HIGH RECALL.

==================================================
10. IMPROVE BLOCKING
==================================================

Investigate additional efficient blocking strategies such as:

- country + exact normalized name
- country + compact name
- country + name prefix
- country + house number
- country + postal code where available
- country + city/locality tokens where extractable
- address token based blocks
- character n-gram retrieval
- other memory-efficient approximate retrieval methods

Use multiple blocking strategies and UNION their candidates.

Do NOT require all blocking conditions simultaneously.

Example:

Candidates =
    exact-name candidates
    UNION
    prefix candidates
    UNION
    address candidates
    UNION
    other high-recall candidates

Be careful about blocks that create millions of candidates.

Measure:

- candidate recall
- average candidates per Source 1
- median candidates
- max candidates
- percentage of Source 1 records with zero candidates

==================================================
11. DO NOT USE EXTERNAL DATA
==================================================

This challenge prohibits external data augmentation.

Do NOT use:

- Google
- LinkedIn
- business databases
- government databases
- geocoding APIs
- external entity lookup APIs
- web search for entity information

Only use the provided challenge data.

==================================================
12. AFTER BLOCKING — FEATURE ENGINEERING
==================================================

Once candidate recall is sufficiently high, create pairwise features.

Potential features:

NAME FEATURES
- exact name match
- normalized name equality
- compact name equality
- character similarity
- Levenshtein/edit similarity
- Jaro/Jaro-Winkler if useful
- token Jaccard
- token overlap
- character n-gram similarity
- TF-IDF cosine similarity

ADDRESS FEATURES
- exact normalized address
- compact address equality
- character similarity
- token overlap
- house-number match
- postal-code match if available
- city/locality overlap
- address token similarity

OTHER
- country equality
- same source indicator
- block type
- name length difference
- address length difference

Do not blindly create huge dense feature matrices.

The dataset is extremely large, so memory efficiency matters.

==================================================
13. NEGATIVE SAMPLING
==================================================

Ground truth gives positive pairs.

Generate negative pairs from candidate pairs that are NOT in the ground truth.

Do NOT train using random negatives only.

Include HARD NEGATIVES such as:

- very similar business names but different addresses
- same house number but different businesses
- same country + similar name but wrong entity
- similar addresses but different businesses

This is important because the challenge has a precision-heavy evaluation.

==================================================
14. MODEL
==================================================

After feature generation, train a pairwise classifier.

Candidate models can include:

- XGBoost
- LightGBM
- CatBoost

Since GPU resources are available, GPU-enabled training can be used if supported.

However, correctness and memory efficiency are more important than simply using GPU.

The model should predict:

P(candidate pair is a true match)

==================================================
15. VALIDATION SPLIT
==================================================

Avoid leakage.

Split validation by Source 1 entity rather than randomly splitting individual pairs.

The same Source 1 entity should not appear in both train and validation pair sets.

Evaluate on unseen Source 1 entities.

==================================================
16. IMPORTANT EVALUATION METRIC
==================================================

The challenge evaluates using macro F0.5 per Source 1 entity.

F0.5 weights precision more heavily than recall.

Therefore:

DO NOT simply accept every vaguely similar candidate.

False merges are costly.

The final threshold must be tuned using validation data.

Also account for entities with ZERO true matches.

A Source 1 entity with no match should be allowed to produce an empty match list.

==================================================
17. MULTIPLE MATCHES
==================================================

Do NOT assume one Source 1 entity has only one match.

The ground truth contains multiple matches.

Therefore final inference should be:

For each Source 1:
    score all candidates
    keep candidates above the selected threshold
    allow zero, one, or many matches

Do NOT simply select:

top 1 candidate

unless validation proves that a separate mechanism is justified.

==================================================
18. FINAL OUTPUT
==================================================

The final system must generate:

output/matching_results.tsv

and:

output/candidate_pairs.tsv

matching_results.tsv should have:

source1_entity_id
matched_entity_ids

There must be exactly one row per Source 1 test entity.

If there are no matches:

matched_entity_ids should be empty.

No duplicate matched IDs.

All matched IDs must come from test Source 2 or Source 3.

candidate_pairs.tsv must contain the candidate IDs actually considered/scored by the final matcher.

Final matches MUST be a subset of candidate pairs.

==================================================
19. VALIDATION
==================================================

Before considering the project complete, run:

python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

Adapt the test directory path if our local project uses data/test instead of dataset/test.

Do not claim the submission is valid until the validator passes.

==================================================
20. CODE ORGANIZATION
==================================================

Please keep the code modular.

Suggested structure:

src/
    preprocessing.py
    blocking.py
    features.py
    train.py
    predict.py
    evaluation.py

notebooks/
    01_eda.ipynb
    02_blocking.ipynb
    03_features.ipynb
    04_training.ipynb
    05_evaluation.ipynb

models/
processed/
output/

Do not put the entire ML pipeline into one huge notebook.

==================================================
21. MEMORY AND PERFORMANCE
==================================================

This is extremely important.

We have millions of records.

Avoid:

- nested Python loops over all Source 1 × Source 2/3
- huge Cartesian products
- storing millions of Python lists unnecessarily
- converting everything to object-heavy structures
- unnecessary dataframe copies
- loading huge intermediate datasets repeatedly

Prefer:

- vectorized Pandas operations
- efficient joins/merges
- indexed retrieval
- categorical/dtype optimization
- chunked processing
- Parquet for large intermediate data if appropriate
- sparse matrices where appropriate
- batch processing
- GPU batches where useful

Before implementing an expensive operation, estimate its memory/time cost.

==================================================
22. GIT RULES
==================================================

Do not directly modify main.

Create your own branch from preprocessing, for example:

gpu-training

Commands:

git checkout preprocessing
git pull origin preprocessing
git checkout -b gpu-training

Commit your work regularly.

Example:

git add src/
git add notebooks/
git commit -m "Add candidate features and training pipeline"

Then:

git push -u origin gpu-training

==================================================
23. DO NOT COMMIT THE DATA
==================================================

Do NOT push:

data/
processed/
models/

unless explicitly required and safe.

The dataset should remain local.

==================================================
24. BEFORE CHANGING EXISTING CODE
==================================================

First inspect:

src/preprocessing.py
src/blocking.py
notebooks/01_eda.ipynb
notebooks/02_blocking.ipynb
utils/validate_submission.py

Understand what they currently do.

If you need to change blocking.py because it is too slow or has low recall:

1. explain why
2. make the improvement
3. test it
4. report candidate recall
5. commit the change

Do not silently replace the existing approach.

==================================================
25. WHAT I EXPECT FROM YOU
==================================================

Work in this order:

PHASE 1
Pull and inspect repository.

PHASE 2
Evaluate current blocking recall.

PHASE 3
Improve blocking for high recall and reasonable candidate volume.

PHASE 4
Create pairwise features.

PHASE 5
Generate positives and hard negatives.

PHASE 6
Train GPU-accelerated matching model.

PHASE 7
Tune decision threshold using macro F0.5.

PHASE 8
Evaluate on validation Source 1 entities.

PHASE 9
Run inference on test data.

PHASE 10
Generate:
    matching_results.tsv
    candidate_pairs.tsv

PHASE 11
Run the official validator.

PHASE 12
Commit and push the completed ML pipeline to your branch.

==================================================
26. REPORT BACK AFTER EACH MAJOR STEP
==================================================

Do not just say "done".

For candidate generation report:

- number of Source 1 records evaluated
- number of candidates
- average candidates/S1
- median candidates/S1
- max candidates/S1
- candidate recall

For model training report:

- number of positive pairs
- number of negative pairs
- number of hard negatives
- features used
- train/validation split
- validation precision
- validation recall
- validation F0.5
- chosen threshold

For final inference report:

- number of test Source 1 entities
- number with zero predicted matches
- number with one predicted match
- number with multiple predicted matches
- total predicted matches
- candidate count
- validator result

==================================================
FINAL PRINCIPLE
==================================================

The main objective is NOT simply to train the most complicated model.

The pipeline must be:

HIGH-RECALL CANDIDATE GENERATION
        ↓
GOOD PAIRWISE FEATURES
        ↓
PRECISION-FOCUSED MATCHING MODEL
        ↓
THRESHOLD TUNING FOR F0.5
        ↓
VALIDATED FINAL OUTPUT

Preserve the existing work, optimize carefully for the millions-of-record scale, and report measurable results after each stage.
