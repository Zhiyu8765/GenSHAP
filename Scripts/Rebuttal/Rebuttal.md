# GenSHAP Rebuttal Audit

## 1. Concern-to-evidence map

| Concern | Shared/severity | Existing evidence | Status |
|---|---|---|---|
| Strict source-level generalization | AC, gcUt, SaUX; critical | Main LR outputs are source-disjoint; four additional 80-train/5-test source-disjoint partitions provide 20 held-out task evaluations, paired with newly trained MLPbV models | **Ready; main and repeated-partition evidence available** |
| Online auxiliary-feature cost | AC, h7vB, gcUt; critical | Logged 0.14–1.45 s totals wrap all deterministic preprocessing, auxiliary extraction, graph/SVD construction, and forward prediction | **Ready; wording and timing breakdown needed** |
| Full offline cost and break-even | AC and all reviewers; critical | GenSHAP optimization: 8,702.46 s, but training preprocessing and label generation are absent. GenSHAP-S rerun: 89,447.45 s after labels, but label generation is absent | **New measurement required** |
| Repeats/error bars and architecture attribution | AC, gcUt; critical | Only single runs exist; architecture variants were trained on `Random_PCA`, not the strict split | **New experiment required** |
| \(R=5000\) Monte Carlo noise floor | AC, h7vB, Z5Z8; critical | Final averaged `PerDShap` values exist, but per-permutation contributions or independent reference replicates do not | **New experiment required** |
| “Compute” versus “approximate” | AC, SaUX, Z5Z8; major | Method learns finite-budget permutation estimates | **Ready; terminology revision required** |
| GenSHAP-S accuracy/cost trade-off | h7vB; major | Main MSE improves on all five tasks, but online time is 60–1,043 s versus 0.14–1.45 s for GenSHAP | **Partial answer; matched-cost experiment required for the direct request** |
| Scalability to large \(N\) | gcUt, Z5Z8; major | Code uses dense pairwise matrices, fully connected graphs, full SVD; current tests are around \(N\leq1000\) | **Scope can be clarified; scaling claim requires experiment** |
| KDE objective clarity | gcUt; moderate | Formula and code use \(w_i\propto\rho_i\), emphasizing dense regions | **Clarification ready; current ablation comparison must be rerun consistently** |
| Generative/multitask/out-of-distribution use | Z5Z8; moderate | No such experiments exist | **Narrow scope; do not claim support** |
| More learned baselines | Z5Z8; moderate | A fresh MLPbV model was trained on the same 80 training tasks in each of four source-disjoint partitions and evaluated on the same held-out tasks as GenSHAP | **Matched MLPbV comparison ready; no additional learned-baseline family is implemented** |
| Public resources | Z5Z8; moderate | Local code/data/models exist, but the submitted anonymous URL is currently inaccessible | **Repository/link fix required** |

## 3. Ready-to-use shared opening

> We thank the AC and reviewers for identifying the central issues of source-level generalization, complete computational accounting, statistical reliability, and the fidelity of our supervision. A repository audit revealed that the main LR results were evaluated under a stricter source-disjoint protocol than Appendix Table 6 indicated: Table 6 accidentally listed the sources of our separate random-PCA robustness corpus rather than the fixed-PCA corpus used for the main results. We provide the corrected split below and will replace the table. We also agree that GenSHAP approximates finite-budget Permutation Data Shapley estimates rather than computing exact Shapley values, and we will revise the title and claims accordingly. Finally, we distinguish below between the full online time already measured in the paper and the offline amortization analysis for which additional measurements are required.

## 4. Generalization

### Draft response to the AC, gcUt, and SaUX

# Source-level generalisation

We thank the reviewers for raising the question of whether the evaluation datasets are genuinely unseen at the source-dataset level. We have verified the source identities directly from the task manifests and provide the explicit source-level split below. The source list in Appendix Table 6 will be updated accordingly. For the fixed-representation LR experiment, the training corpus contains 100 constructed valuation tasks derived from 33 source datasets. The evaluation corpus contains six tasks derived from five held-out sources: Artificial Characters, Iris, STL10, Vehicle, and Volkert. None of these five sources occurs in the training corpus:
\[
\mathcal{S}_{\mathrm{train}}\cap\mathcal{S}_{\mathrm{test}}=\varnothing.
\]
The model is therefore trained once while all five evaluation sources are excluded simultaneously. This evaluates transfer to unseen sources, rather than transfer between different subsets or class selections derived from the same source. No samples, labels, graphs, or reference Shapley estimates from the held-out sources are used to optimise GenSHAP. To provide a fresh verification under this explicit protocol, we trained the full GenSHAP model from scratch using three independent seeds (23, 42, and 2026). All runs used the same 100-task training manifest and the same pre-specified hyperparameters. In each run, the final-epoch checkpoint was saved before the held-out data were loaded; hence, the held-out results were not used for optimisation or checkpoint selection. We report the corresponding mean and sample standard deviation in the comparison table below. These experiments support a deliberately scoped conclusion: GenSHAP transfers to source-disjoint held-out classification datasets under the evaluated LR utility, fixed-representation preprocessing, and dataset-size regime. We will revise broader wording such as “any unseen classification dataset” to reflect this empirical scope.

## Updated source-level dataset list

The table reports source datasets rather than individual constructed task filenames. “Tasks” denotes the number of paired valuation tasks derived from each source. “Saved dim.” is the feature dimension in the task files before GenSHAP constructs its relation-based representation. The fixed preprocessing retains the original dimension when it is at most 20 and reduces higher-dimensional inputs to 20 dimensions.

| Source dataset | Use | Tasks | Saved dim. |
|---|---|---:|---:|
| Autos | Training | 3 | 20 |
| Bank Marketing | Training | 3 | 20 |
| Breast Cancer | Training | 2 | 20 |
| Caltech101 | Training | 3 | 20 |
| CIFAR-10 | Training | 3 | 20 |
| CIFAR-100 | Training | 3 | 20 |
| Connect-4 | Training | 3 | 20 |
| Covertype | Training | 3 | 20 |
| CovPokElec | Training | 3 | 20 |
| Credit Score | Training | 3 | 20 |
| Devnagari-Script | Training | 3 | 20 |
| Dilbert | Training | 3 | 20 |
| Dry Bean | Training | 3 | 16 |
| EEG Eye State | Training | 3 | 14 |
| FashionMNIST | Training | 3 | 20 |
| Gas Drift | Training | 3 | 20 |
| HAR | Training | 3 | 20 |
| hls4ml LHC Jets HLF | Training | 3 | 16 |
| ImageNet-A | Training | 3 | 20 |
| Jannis | Training | 3 | 20 |
| Kuzushiji-MNIST | Training | 3 | 20 |
| LED(50000) | Training | 3 | 20 |
| Letter | Training | 3 | 16 |
| MNIST | Training | 5 | 20 |
| Mozilla4 | Training | 3 | 5 |
| Nomao | Training | 3 | 20 |
| Pendigits | Training | 3 | 16 |
| Pokerhand | Training | 3 | 10 |
| RandomRBF 0 0 | Training | 3 | 10 |
| RT-IoT2022 | Training | 3 | 20 |
| SDSS17 | Training | 3 | 20 |
| Shuttle | Training | 3 | 9 |
| Spoken Arabic Digit | Training | 3 | 14 |
| Artificial Characters | Unseen test | 1 | 7 |
| Iris | Unseen test | 1 | 4 |
| STL10 | Unseen test | 1 | 20 |
| Vehicle | Unseen test | 1 | 18 |
| Volkert | Unseen test | 2 | 20 |

Training total: 100 tasks from 33 sources.  
Unseen-test total: six tasks from five sources.  
Source intersection: empty.

## Updated comparison with baselines

The table reports MSE against the high-budget Permutation Data Shapley reference to eight decimal places. Baseline entries are the results reported in the submission. The updated GenSHAP entry is the mean \(\pm\) sample standard deviation over the three new source-disjoint runs. For direct correspondence with the submitted baseline table, the Volkert column uses the `volkert_15_2` task.

| TMC Shapley | 0.00005100 | 0.00000095 | 0.00000041 | 0.00000022 | 0.00000038 |
| Delta Shapley | 0.00013700 | 0.00000408 | 0.00000083 | 0.00000045 | 0.00000081 |
| Group-Testing Shapley | 0.03945400 | 0.00556213 | 0.00185878 | 0.00121310 | 0.00306043 |
| Owen Shapley | 0.00000300 | 0.00000013 | 0.00000004 | 0.00000002 | 0.00000007 |
| MLPbV | 0.00012900 | 0.00012055 | 0.00008788 | 0.00008770 | 0.00006617 |
| GenSHAP (source-disjoint, 3 runs) | \(0.00023267\pm0.00006636\) | \(0.00000449\pm0.00000036\) | \(0.00000122\pm0.00000018\) | \(0.00000052\pm0.00000004\) | \(0.00000063\pm0.00000005\) |

The classical baselines compute values separately for each target dataset and therefore do not use the cross-source training protocol. This table compares approximation fidelity; the source-disjoint manifest above establishes the generalisation protocol for GenSHAP. The results should be described as an accuracy-efficiency trade-off rather than a universal MSE improvement: high-budget TMC and Owen obtain lower MSE, whereas GenSHAP amortises valuation across datasets and has substantially lower online cost. Among learned estimators, GenSHAP remains substantially more accurate than MLPbV on Vehicle, Characters, STL10, and Volkert, while MLPbV is better on Iris.

### Additional source-split generalisation evaluation

To evaluate whether the observed generalisation depends on one particular train/test division, we conducted an additional four-partition source-disjoint repeated-holdout experiment. This complements the original fixed five-source evaluation and, unlike the earlier draft of this audit, now includes a matched learned-baseline comparison with MLPbV.

#### Experimental protocol

We merged the available fixed-representation LR tasks into a pool containing 106 complete valuation tasks from 38 source datasets. We constructed three deterministic random partitions using seeds 23, 42, and 2026. We then constructed a fourth deterministic partition using seed 20260727, with Iris pre-specified as a held-out source and the other four held-out sources selected by the seeded procedure. Each partition contains:

- 80 randomly selected training tasks;
- five test tasks from five distinct held-out source datasets; and
- zero source overlap between its training and test tasks.

If one task from a source is selected for testing, every other task derived from that source is excluded from that partition’s training set. The held-out source groups are non-overlapping across the first three partitions. The fourth partition repeats Autos, Iris, and Kuzushiji-MNIST under a different 80-task training set and adds Connect-4 and Pendigits. The experiment therefore contains 20 held-out task evaluations covering 17 distinct source datasets:

| Split | Held-out source datasets |
|---|---|
| 1 | Autos, CIFAR-100, hls4ml LHC Jets HLF, ImageNet-A, Iris |
| 2 | Bank Marketing, Covertype, Gas Drift, HAR, Kuzushiji-MNIST |
| 3 | Caltech101, EEG Eye State, Mozilla4, RT-IoT2022, Volkert |
| 4 | Autos, Connect-4, Iris, Kuzushiji-MNIST, Pendigits |

For each partition, we trained one full GenSHAP model using the same architecture and pre-specified hyperparameters as the main LR experiment. Model seed 23 was fixed across partitions so that differences primarily reflect the source partition rather than neural initialization. Each final-epoch checkpoint was saved before its held-out tasks were loaded, and no test-based checkpoint or hyperparameter selection was performed. All four runs completed successfully and produced separate checkpoints. For the fourth run, the saved metadata records 80 training tasks, the five expected holdouts, and a checkpoint SHA-256 matching the saved model.

For the learned-baseline comparison, we independently trained one MLPbV model on the same 80 training tasks in each partition, using the fixed MLPbV implementation and hyperparameters used for the submitted baseline. Each MLPbV model was then evaluated only on the corresponding five held-out tasks. Thus, the GenSHAP–MLPbV results below are paired by both training partition and held-out task.

This protocol should be described as a **four-partition source-disjoint repeated-holdout evaluation**, rather than classical exhaustive \(k\)-fold cross-validation, because each partition samples 80 training tasks instead of assigning every task to exactly one fold.

#### Paired per-source MSE results

| Split | Held-out source | Valued points | GenSHAP MSE | MLPbV MSE | Lower MSE |
|---:|---|---:|---:|---:|---|
| 1 | Autos | 965 | 0.00000035 | 0.00005400 | GenSHAP |
| 1 | CIFAR-100 | 500 | 0.00000280 | 0.00005742 | GenSHAP |
| 1 | hls4ml LHC Jets HLF | 906 | 0.00000048 | 0.00003402 | GenSHAP |
| 1 | ImageNet-A | 959 | 0.00000114 | 0.00010204 | GenSHAP |
| 1 | Iris | 75 | 0.00011188 | 0.00012522 | GenSHAP |
| 2 | Bank Marketing | 1,000 | 0.00000075 | 0.00007656 | GenSHAP |
| 2 | Covertype | 1,000 | 0.00000042 | 0.00004288 | GenSHAP |
| 2 | Gas Drift | 1,000 | 0.00000024 | 0.00006758 | GenSHAP |
| 2 | HAR | 1,000 | 0.00000018 | 0.00007872 | GenSHAP |
| 2 | Kuzushiji-MNIST | 906 | 0.00000030 | 0.00006858 | GenSHAP |
| 3 | Caltech101 | 964 | 0.00000028 | 0.00008577 | GenSHAP |
| 3 | EEG Eye State | 1,000 | 0.00000150 | 0.00007006 | GenSHAP |
| 3 | Mozilla4 | 1,000 | 0.00000096 | 0.00013034 | GenSHAP |
| 3 | RT-IoT2022 | 928 | 0.00000041 | 0.00009634 | GenSHAP |
| 3 | Volkert | 995 | 0.00000058 | 0.00005749 | GenSHAP |
| 4 | Autos | 995 | 0.00000039 | 0.00005136 | GenSHAP |
| 4 | Connect-4 | 1,000 | 0.00000051 | 0.00007811 | GenSHAP |
| 4 | Iris | 75 | 0.00027137 | 0.00013849 | MLPbV |
| 4 | Kuzushiji-MNIST | 969 | 0.00000036 | 0.00014607 | GenSHAP |
| 4 | Pendigits | 1,000 | 0.00000023 | 0.00010514 | GenSHAP |

GenSHAP has lower MSE on 19 of the 20 paired held-out evaluations. The exception is Iris in Split 4, where MLPbV obtains \(0.00013849\) and GenSHAP obtains \(0.00027137\). We retain this exception explicitly rather than implying uniform dominance.

The raw MSE scale varies substantially across datasets because individual Shapley values generally become smaller as the number of valued points increases. In particular, each Iris evaluation contains only 75 valued points and has much larger reference-value magnitudes than the tasks containing approximately 1,000 points. The two Iris evaluations therefore dominate the raw GenSHAP macro MSE. We consequently report the median, point-weighted, and energy-normalized results alongside the task-macro average.

#### Matched learned-baseline summary

| Split | GenSHAP macro MSE | MLPbV macro MSE | MLPbV / GenSHAP | GenSHAP wins |
|---:|---:|---:|---:|---:|
| 1 | 0.00002333 | 0.00007454 | \(3.20\times\) | 5/5 |
| 2 | 0.00000038 | 0.00006686 | \(176.32\times\) | 5/5 |
| 3 | 0.00000075 | 0.00008800 | \(117.88\times\) | 5/5 |
| 4 | 0.00005457 | 0.00010383 | \(1.90\times\) | 4/5 |
| **All 20 evaluations** | **0.00001976** | **0.00008331** | **\(4.22\times\)** | **19/20** |

| Aggregate metric | GenSHAP | MLPbV |
|---|---:|---:|
| Task-macro MSE | 0.00001976 | 0.00008331 |
| Median task MSE | 0.00000045 | 0.00007733 |
| Point-weighted micro MSE | 0.00000227 | 0.00007906 |
| Macro energy-normalized MSE | 0.4947 | 56.0567 |
| Macro Spearman correlation | 0.2159 | 0.0425 |
| Evaluations outperforming the zero predictor | 19/20 | 2/20 |
| Taskwise MSE wins | 19/20 | 1/20 |

Across the 20 paired evaluations, GenSHAP reduces task-macro MSE by \(76.28\%\) relative to MLPbV. Its median task MSE is approximately \(172\) times lower. These paired results extend the learned-baseline comparison beyond the original five fixed test sources while using identical meta-level training and test partitions for both learned methods.

#### Scale-aware fidelity

We define the energy-normalized MSE for held-out task \(d\) as

\[
\operatorname{ENMSE}_d
=
\frac{
\frac{1}{N_d}\sum_{i=1}^{N_d}
(\widehat{\phi}_{di}-\phi_{di})^2
}{
\frac{1}{N_d}\sum_{i=1}^{N_d}\phi_{di}^{\,2}
}.
\]

This normalization accounts for differences in the magnitude of the reference Shapley values. It also has a direct interpretation: a trivial predictor that assigns zero to every point has \(\operatorname{ENMSE}=1\). Therefore, values below one indicate that GenSHAP captures nontrivial information about the held-out reference values.

| Split | GenSHAP macro MSE | GenSHAP median MSE | GenSHAP macro ENMSE | Error reduction versus zero | Range-normalized RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.00002333 | 0.00000114 | 0.519053 | 48.09% | 21.03% |
| 2 | 0.00000038 | 0.00000030 | 0.329711 | 67.03% | 9.96% |
| 3 | 0.00000075 | 0.00000058 | 0.515836 | 48.42% | 15.29% |
| 4 | 0.00005457 | 0.00000039 | 0.614368 | 38.56% | 28.55% |
| **All 20 evaluations** | **0.00001976** | **0.00000045** | **0.494742** | **50.53%** | **18.71%** |

Nineteen of the 20 GenSHAP evaluations have an energy-normalized MSE below one; the exception is Iris in Split 4, with ENMSE \(1.544883\). Averaged equally across evaluations, GenSHAP’s MSE is 49.47% of the error produced by the zero-value predictor, corresponding to an average error reduction of 50.53%. The median energy-normalized MSE is 0.403546, and the median range-normalized RMSE is 12.49%.

These normalized results show that the aggregate advantage is not obtained merely because the reference values of large datasets are numerically close to zero. They also expose the one partition-task combination on which GenSHAP does not outperform the zero predictor.

#### Ranking and low-value recovery

Because data valuation is frequently used to rank or remove training points, we additionally evaluated scale-independent ranking metrics from the saved GenSHAP predictions. The mean Spearman correlation across the 20 held-out evaluations is 0.215857, with positive correlation on 18 of 20 evaluations.

For a held-out task \(d\), BottomRecall@10 measures the overlap between the predicted and reference bottom-10% sets. ExtremeRecall@10 averages the corresponding bottom-10% and top-10% recalls. A random ranking has an expected recall of 10% for either metric.

| Split | Mean Spearman \(\rho\) | BottomRecall@10 | ExtremeRecall@10 |
|---:|---:|---:|---:|
| 1 | 0.216845 | 33.39% | 19.65% |
| 2 | 0.247096 | 36.58% | 24.88% |
| 3 | 0.229638 | 27.84% | 21.96% |
| 4 | 0.169851 | 33.16% | 21.95% |
| **All 20 evaluations** | **0.215857** | **32.74%** | **22.11%** |
| Random-ranking expectation | — | 10.00% | 10.00% |

Across all four source partitions, the predicted rankings recover 32.74% of the genuinely lowest-valued points, more than three times the random-ranking expectation. The combined extreme-set recall is 22.11%, compared with the 10% random expectation. These results provide additional evidence that GenSHAP transfers operationally useful ranking information to unseen sources, particularly for identifying low-value or potentially harmful observations.

#### Interpretation and relationship to the baseline comparison

We do not use these additional partitions to claim superiority over the per-target classical sampling baselines because their outputs are not available for these 20 task evaluations. The experiment instead addresses whether a learned GenSHAP model continues to capture nontrivial Shapley-value information when the held-out source identities change, and it now supplies a controlled comparison against another learned estimator trained on exactly the same partitions.

The original five-source experiment remains the controlled comparison with the full submitted baseline set, including TMC Shapley, Owen Shapley, and MLPbV. The new experiment broadens the repeated-partition generalisation assessment to 20 evaluations covering 17 distinct held-out sources. Together with the original five-source split, 20 distinct source datasets have served as held-out sources across our experiments.

The evidence supports a deliberately scoped conclusion: under the evaluated LR utility, fixed-representation preprocessing, and dataset-size regime, GenSHAP’s transfer is not restricted to one favorable source partition and is substantially more accurate than MLPbV on 19 of 20 paired evaluations. Nevertheless, the Split 4 Iris result demonstrates that transfer is not uniformly successful, and this is not evidence of universal transfer to arbitrary domains, base learners, or dataset scales. We will therefore retain the narrowed generalisation claim.

## 5. Online end-to-end timing: completed answer

### Draft response to h7vB W1 and gcUt

We apologize that “single forward pass” made the accounting ambiguous. The reported GenSHAP times of 0.14–1.45 s are already full **online wall-clock times**, not forward-only times. The timer starts before the target-dataset preprocessing routine and includes data loading, the pairwise distance matrix, KNN reliability features, fitting the full LR utility model, gradient statistics, the 21 learning-dynamics fits at iterations \(1,10,\ldots,200\), metadata, all five graph/SVD constructions, device synchronization, and the final neural prediction. The forward prediction alone is below 0.03 s. We will replace “single-forward-pass inference time” with two explicit columns, “target preprocessing and representation” and “neural forward,” and define their sum as “online total.” The measured decomposition is shown below. This directly addresses the concern that the auxiliary extraction might add an unreported 30–60 s for GenSHAP: it is already included in the first column.

| Dataset | GenSHAP target preprocessing/representation (s) | Forward (s) | Online total (s) |
|---|---:|---:|---:|
| Iris | 0.117571 | 0.017303 | 0.134874 |
| Vehicle | 0.515252 | 0.021982 | 0.537234 |
| Characters | 1.088560 | 0.029022 | 1.117582 |
| STL10 | 1.423204 | 0.028722 | 1.451926 |
| Volkert | 1.356950 | 0.028975 | 1.385925 |

The same decomposition for the originally reported GenSHAP-S run is:

| Dataset | Sampled preprocessing/representation (s) | Forward (s) | Online total (s) |
|---|---:|---:|---:|
| Iris | 60.100487 | 0.015879 | 60.116366 |
| Vehicle | 551.786207 | 0.020430 | 551.806637 |
| Characters | 793.645424 | 0.027387 | 793.672811 |
| STL10 | 1042.932964 | 0.027924 | 1042.960888 |
| Volkert | 770.847220 | 0.027501 | 770.874721 |


### Matched-hardware end-to-end runtime. 
We thank the reviewers for identifying that our original runtime comparison used different hardware: the classical baselines were evaluated on CPUs, whereas the submitted GenSHAP measurements used an RTX 3090. To remove this confound, we re-evaluated a frozen full GenSHAP model with GPU access disabled on the same dual-socket Intel Xeon E5-2678 v3 server used for the baseline experiments (24 physical cores, 48 logical CPUs, and 251 GiB memory). We used one untimed warm-up followed by ten timed evaluations for each held-out dataset. The timer starts before loading and processing the target task and includes all target-specific online computation: data loading, pairwise-distance and KNN-feature calculation, fitting the full LR utility model, gradient and learning-dynamics features, construction of all five graphs and their SVD representations, and the final neural prediction. Thus, the reported values are complete online wall-clock times rather than neural-forward-only times.
| Dataset | Target preprocessing and representation (s) | Neural forward (s) | Online total (s) |
|---|---:|---:|---:|
| Iris | \(0.139623 \pm 0.022654\) | \(0.019398 \pm 0.003319\) | \(0.159022 \pm 0.022938\) |
| Vehicle | \(0.523118 \pm 0.006391\) | \(0.049905 \pm 0.000489\) | \(0.573025 \pm 0.006423\) |
| Artificial Characters | \(1.101133 \pm 0.055448\) | \(0.173108 \pm 0.019735\) | \(1.274242 \pm 0.060005\) |
| STL10 | \(1.135885 \pm 0.018986\) | \(0.175116 \pm 0.009964\) | \(1.311003 \pm 0.025251\) |
| Volkert | \(1.245672 \pm 0.035988\) | \(0.181162 \pm 0.009684\) | \(1.426835 \pm 0.042651\) |
All entries are means \(\pm\) sample standard deviations over ten timed runs. The one-time checkpoint-loading cost was \(0.077221\) s and is reported separately rather than charged to every target dataset. Even if this loading cost were conservatively charged in full to every evaluation, the resulting totals would remain between \(0.236243\) s and \(1.504056\) s.
The matched-CPU results show that GenSHAP’s online totals remain between \(0.159022\) s and \(1.426835\) s across the five held-out datasets. Most of this time—approximately 86%–91%—is spent on target-specific preprocessing and representation rather than GPU computation. Therefore, the low online runtime is not an artifact of evaluating GenSHAP on an RTX 3090. We will replace the ambiguous phrase “single-forward-pass inference time” with the explicit decomposition above and use these CPU-only measurements in the comparison with the CPU baseline results.




## 6. Matched sampling-budget comparison for GenSHAP-S
We thank the reviewer for asking whether GenSHAP-S’s improvement merely results from its additional target-specific sampling. GenSHAP-S draws 100 coalitions for every valued point. Because each marginal probe evaluates both \(U(S)\) and \(U(S\cup\{i\})\), this corresponds to a nominal sampling budget of \(200N\) LR utility fits for a dataset containing \(N\) valued points.
We therefore evaluated TMC-Shapley and Owen-Shapley under the same nominal \(200N\) sampling-related LR-fit allowance on the five held-out LR tasks. TMC received 200 permutations and retained its standard relative-truncation mechanism, while Owen received 100 paired marginal samples per point. We used the same LR utility and held-out accuracy metric. GenSHAP-S achieves lower MSE than budget-capped TMC on three of five datasets and lower MSE than matched-budget Owen on two of five datasets. No method dominates across all tasks: GenSHAP-S performs best on Characters and Volkert, TMC performs best on Iris, and Owen performs best on Vehicle and STL10.
TMC’s truncation used only 35.49%–51.78% of its allocated budget, while Owen used 98.70%–99.00%. We retain this computational advantage rather than forcing TMC to perform unnecessary fits. These results therefore support a dataset-dependent accuracy–cost trade-off rather than universal superiority. We will position full GenSHAP as the speed-oriented method and GenSHAP-S as an optional accuracy-oriented variant when additional target-specific computation is acceptable.

| Method | Iris | Vehicle | Characters | STL10 | Volkert |
|---|---:|---:|---:|---:|---:|
| TMC-Shapley | **0.00003295** | 0.00000162 | 0.00000095 | 0.00000054 | 0.00000176 |
| Owen-Shapley | 0.00005695 | **0.00000147** | 0.00000045 | **0.00000029** | 0.00000053 |
| GenSHAP-S | 0.00008299 | 0.00000221 | **0.00000041** | 0.00000033 | **0.00000037** |



## 7. Complete offline cost and amortization
We thank the reviewers for requesting complete computational accounting. We distinguish the one-time offline cost of constructing and training GenSHAP from the online cost of valuing a new target dataset.
The reference supervision was generated using \(R=5000\) untruncated permutations per training task. Under the same LR implementation, hardware, and parallelism, generating labels for a task with approximately 1,000 valued points required approximately 2–3 hours. Our training corpus contains 100 tasks and 95,107 valued points; 94 of the 100 tasks contain at least 900 points. We therefore conservatively estimate the aggregate reference-label-generation cost as approximately 200–300 serial computation hours. Because individual per-task timing files from the original label-generation runs were not retained, we report this component as a range rather than with spurious precision. Parallelizing tasks reduces elapsed wall-clock time but not this aggregate computation.
We additionally performed a timing-only run of the complete deterministic training preprocessing for all 100 fixed-PCA training tasks. This experiment called the same load_and_preprocess routine used by GenSHAP training, including data loading, pairwise distances, KNN reliability features, the full LR utility fit, gradient features, the 21 learning-dynamics LR fits, and construction of all five graph/SVD representations. It did not instantiate or optimize the neural network and did not modify labels, checkpoints, or predictions. The run was conducted on the same dual-socket Intel Xeon E5-2678 v3 server with an RTX 3090 visible for graph construction.
| Offline component | Cost |
|---|---:|
| Reference-label generation, \(R=5000\) | approximately 200–300 h |
| Training preprocessing, 100 tasks | 129.252274 s (0.035903 h) |
| Neural-network optimization | 8,702.459291 s (2.417350 h) |
| Post-label GenSHAP subtotal | 8,831.711565 s (2.453253 h) |
| Complete GenSHAP offline cost | approximately 202.453–302.453 h |
The preprocessing experiment covered all 100 tasks and 95,107 valued points. The mean and median preprocessing times were 1.292523 s and 1.346962 s per task, respectively; the full instrumented process, including timing-output bookkeeping and input hashing, took 143.359317 s.
For completeness, the corresponding measured post-label cost of GenSHAP-S is 89,447.450387 s (24.846514 h), comprising 80,754.554173 s for sampled-probe and graph construction and 8,692.896214 s for neural optimization. Including the same reference supervision gives an estimated complete GenSHAP-S offline cost of approximately 224.847–324.847 h. We therefore position full GenSHAP as the amortized, speed-oriented method and GenSHAP-S as an optional accuracy-oriented variant with additional computation.
Explicit break-even analysis
We also make the amortization claim explicit. For baseline \(b\), the number of future valuation tasks required to amortize the complete offline cost is
\[
K_b^\star
=
\left\lceil
\frac{C_{\mathrm{offline}}}
     {\bar{T}_b-\bar{T}_{\mathrm{GenSHAP}}}
\right\rceil ,
\]
where the bars denote the average matched-hardware online time over the five held-out task profiles.
Using CPU-only GenSHAP inference and the sampling-budget-matched baselines on the same Xeon server, the average online times were 0.948825 s for GenSHAP, 347.561356 s for budget-capped TMC Shapley, and 861.637564 s for Owen Shapley. Under this five-dataset workload profile, the complete from-scratch cost is amortized after approximately 2,103–3,142 future tasks relative to TMC and 847–1,266 future tasks relative to Owen. If reference labels have already been generated and only the measured 2.453253 h post-label cost is amortized, the corresponding break-even points are 26 tasks and 11 tasks, respectively.
These development-level break-even figures include the historical supervision cost and should not be confused with the deployment cost for users of a released checkpoint. With the trained checkpoint and preprocessing code released, users do not repeat label generation or neural training; they incur only the measured online cost, so the online saving applies from the first target dataset. We will revise the paper to distinguish these two settings and will not claim that GenSHAP reduces total computation for a small number of tasks when training entirely from scratch.


## 8. Ablation evaluation beyond MSE

We thank the reviewer for noting that pointwise mean squared error (MSE) alone does not fully characterize data-valuation quality. This is particularly important because Shapley values are often used to rank data points or to identify the most beneficial and harmful examples. We therefore extended the ablation to four architectures:

- **Full GenSHAP:** the complete model combining the GIN, Transformer, and auxiliary-feature branches;
- **GIN-only:** the graph branch without the Transformer or auxiliary-feature branch;
- **Transformer-only:** the Transformer branch without the GIN or auxiliary-feature branch;
- **MLP-only:** an MLP operating only on the auxiliary features.

Each architecture was trained independently with three random seeds, \(23\), \(42\), and \(2026\), using the same source-isolated training split and evaluated on the same five held-out datasets. The reported values are means \(\pm\) sample standard deviations across the three independently trained models. No test-set-based checkpoint selection was performed.

### Pointwise MSE

For completeness, the repeated-run MSE results are:

| Dataset | Full GenSHAP | GIN-only | Transformer-only | MLP-only |
|---|---:|---:|---:|---:|
| Iris | \(0.00023267 \pm 0.00006636\) | \(0.00006452 \pm 0.00003368\) | \(0.00013357 \pm 0.00007650\) | **\(0.00005717 \pm 0.00001656\)** |
| Vehicle | \(0.00000449 \pm 0.00000036\) | \(0.00000315 \pm 0.00000034\) | **\(0.00000287 \pm 0.00000024\)** | \(0.00000330 \pm 0.00000015\) |
| Artificial Characters | \(0.00000122 \pm 0.00000018\) | **\(0.00000092 \pm 0.00000022\)** | \(0.00000109 \pm 0.00000011\) | \(0.00000105 \pm 0.00000004\) |
| STL10 | **\(0.00000052 \pm 0.00000004\)** | \(0.00000062 \pm 0.00000003\) | \(0.00000055 \pm 0.00000007\) | \(0.00000053 \pm 0.00000002\) |
| Volkert | \(0.00000063 \pm 0.00000005\) | \(0.00000072 \pm 0.00000005\) | \(0.00000065 \pm 0.00000005\) | **\(0.00000057 \pm 0.00000005\)** |
| Macro mean | \(0.00004791\) | \(0.00001399\) | \(0.00002775\) | **\(0.00001252\)** |
| Average method rank \(\downarrow\) | \(3.0\) | \(2.6\) | \(2.6\) | **\(1.8\)** |

No architecture uniformly minimizes MSE: MLP-only is best on Iris and Volkert, Transformer-only on Vehicle, GIN-only on Artificial Characters, and full GenSHAP on STL10. The macro MSE is dominated by Iris because the absolute scale of Shapley values, and consequently the MSE scale, differs substantially across datasets. We therefore also report scale-invariant ranking and downstream utility metrics below.

### Global rank correlation

We first measure agreement between the complete predicted and reference rankings using Spearman’s \(\rho\) and Kendall’s \(\tau\):

| Method | Spearman’s \(\rho\) | Kendall’s \(\tau\) |
|---|---:|---:|
| Full GenSHAP | \(0.215449\) | \(0.146417\) |
| GIN-only | \(0.128243\) | \(0.089846\) |
| Transformer-only | \(0.208658\) | \(0.144664\) |
| MLP-only | **\(0.223915\)** | **\(0.153107\)** |

The MLP-only model gives the highest global correlation, although its advantage over full GenSHAP is small: \(0.008466\) in Spearman correlation and \(0.006690\) in Kendall correlation. Transformer-only is also close to full GenSHAP, whereas GIN-only has the weakest global rank agreement. Thus, the additional branches do not produce a uniform advantage when every point in the ranking is weighted equally.

However, global rank correlation can understate performance in the practically important tails of the valuation distribution. In data valuation, the highest-valued points are candidates for preservation or acquisition, while the lowest-valued points are candidates for removal. We therefore additionally evaluate recovery of these extreme sets.

### Recovery of high- and low-value points

Let \(T_q^{+}\) and \(T_q^{-}\) denote the reference top-\(q\%\) and bottom-\(q\%\) sets, and let \(\widehat{T}_q^{+}\) and \(\widehat{T}_q^{-}\) denote their predicted counterparts. We define

\[
\operatorname{ExtremeRecall@}q
=
\frac{1}{2}
\left(
\frac{|T_q^{+}\cap\widehat{T}_q^{+}|}{|T_q^{+}|}
+
\frac{|T_q^{-}\cap\widehat{T}_q^{-}|}{|T_q^{-}|}
\right).
\]

At \(q=10\%\), the results are:

| Dataset | Full GenSHAP | GIN-only | Transformer-only | MLP-only |
|---|---:|---:|---:|---:|
| Iris | **\(0.125000 \pm 0.000000\)** | \(0.062500 \pm 0.108253\) | \(0.020833 \pm 0.036084\) | \(0.062500 \pm 0.062500\) |
| Vehicle | \(0.170543 \pm 0.024205\) | \(0.158915 \pm 0.026854\) | \(0.166667 \pm 0.046994\) | **\(0.174419 \pm 0.050685\)** |
| Artificial Characters | \(0.305000 \pm 0.069282\) | \(0.271667 \pm 0.120035\) | **\(0.316667 \pm 0.035473\)** | \(0.286667 \pm 0.046458\) |
| STL10 | \(0.225086 \pm 0.033140\) | \(0.156357 \pm 0.011904\) | **\(0.235395 \pm 0.028390\)** | \(0.214777 \pm 0.012972\) |
| Volkert | **\(0.295000 \pm 0.026458\)** | \(0.215000 \pm 0.055678\) | **\(0.295000 \pm 0.031225\)** | \(0.291667 \pm 0.041633\) |
| Macro mean | **\(0.224126\)** | \(0.172888\) | \(0.206912\) | \(0.206006\) |
| Average method rank \(\downarrow\) | **\(1.7\)** | \(3.7\) | \(2.1\) | \(2.5\) |

Full GenSHAP achieves the highest macro ExtremeRecall@10 and the best average rank. Its macro recall is \(29.64\%\) higher than GIN-only, \(8.32\%\) higher than Transformer-only, and \(8.80\%\) higher than MLP-only. Transformer-only performs especially well on Artificial Characters and STL10 and ties full GenSHAP on Volkert, showing that the Transformer branch contributes strongly to identifying extreme-valued points.

Because removal of harmful data is a central downstream use case, we also report bottom-set recall separately:

| Dataset | Full GenSHAP | GIN-only | Transformer-only | MLP-only |
|---|---:|---:|---:|---:|
| Iris | **\(0.250000 \pm 0.000000\)** | \(0.125000 \pm 0.216506\) | \(0.041667 \pm 0.072169\) | \(0.125000 \pm 0.125000\) |
| Vehicle | **\(0.255814 \pm 0.023256\)** | \(0.155039 \pm 0.149514\) | **\(0.255814 \pm 0.061529\)** | \(0.240310 \pm 0.074757\) |
| Artificial Characters | \(0.363333 \pm 0.101160\) | **\(0.463333 \pm 0.275379\)** | \(0.336667 \pm 0.090738\) | \(0.320000 \pm 0.010000\) |
| STL10 | **\(0.319588 \pm 0.035712\)** | \(0.240550 \pm 0.021460\) | \(0.271478 \pm 0.031495\) | \(0.243986 \pm 0.039030\) |
| Volkert | \(0.453333 \pm 0.035119\) | \(0.380000 \pm 0.115326\) | **\(0.473333 \pm 0.035119\)** | \(0.443333 \pm 0.025166\) |
| Macro mean | **\(0.328414\)** | \(0.272784\) | \(0.275792\) | \(0.274526\) |
| Average method rank \(\downarrow\) | **\(1.5\)** | \(3.1\) | \(2.3\) | \(3.1\) |

Full GenSHAP obtains the highest macro BottomRecall@10 and the best average rank. Relative to the three ablations, its macro recall is \(20.39\%\) higher than GIN-only, \(19.08\%\) higher than Transformer-only, and \(19.63\%\) higher than MLP-only.

These conclusions are not specific to the \(10\%\) threshold:

| Metric | Full GenSHAP | GIN-only | Transformer-only | MLP-only |
|---|---:|---:|---:|---:|
| BottomRecall@5 | **\(0.276677\)** | \(0.251017\) | \(0.241828\) | \(0.257955\) |
| BottomRecall@10 | **\(0.328414\)** | \(0.272784\) | \(0.275792\) | \(0.274526\) |
| BottomRecall@20 | **\(0.430271\)** | \(0.360451\) | \(0.404383\) | \(0.384321\) |
| ExtremeRecall@5 | **\(0.180353\)** | \(0.144526\) | \(0.170244\) | \(0.176118\) |
| ExtremeRecall@10 | **\(0.224126\)** | \(0.172888\) | \(0.206912\) | \(0.206006\) |
| ExtremeRecall@20 | \(0.327683\) | \(0.273031\) | **\(0.328919\)** | \(0.315200\) |

Full GenSHAP is best at all three evaluated bottom-set cutoffs and at the \(5\%\) and \(10\%\) extreme-set cutoffs. Transformer-only is narrowly better at ExtremeRecall@20, by \(0.001236\). The principal advantage of the full architecture is therefore concentrated in identifying the most harmful observations, rather than producing uniformly lower coordinate-wise error.

### Downstream removal utility

We next evaluate whether the predicted values lead to the expected behavior when used for data removal. For removal fractions

\[
f\in\{0,0.05,0.10,\ldots,0.40\},
\]

we remove either the lowest-valued or highest-valued points according to each method and refit the LR utility model. Let \(\operatorname{AUC}_{\mathrm{low}}\) and \(\operatorname{AUC}_{\mathrm{high}}\) denote the normalized areas under these two accuracy curves. We report

\[
\operatorname{SeparationAUC}
=
\operatorname{AUC}_{\mathrm{low}}
-
\operatorname{AUC}_{\mathrm{high}},
\]

where a larger value indicates that the method more successfully distinguishes beneficial from harmful training points.

| Dataset | Full GenSHAP | GIN-only | Transformer-only | MLP-only |
|---|---:|---:|---:|---:|
| Iris | **\(-0.008889 \pm 0.012946\)** | \(-0.085556 \pm 0.084530\) | \(-0.023056 \pm 0.026827\) | \(-0.015556 \pm 0.005092\) |
| Vehicle | **\(-0.001970 \pm 0.024164\)** | \(-0.003004 \pm 0.009183\) | \(-0.006501 \pm 0.002236\) | \(-0.010638 \pm 0.018633\) |
| Artificial Characters | \(0.047187 \pm 0.024141\) | **\(0.099146 \pm 0.012428\)** | \(0.032396 \pm 0.019797\) | \(0.039792 \pm 0.004203\) |
| STL10 | **\(0.024270 \pm 0.010415\)** | \(0.002529 \pm 0.013384\) | \(0.010979 \pm 0.007843\) | \(0.017116 \pm 0.008058\) |
| Volkert | **\(0.026214 \pm 0.013442\)** | \(0.020561 \pm 0.024658\) | \(0.008019 \pm 0.009720\) | \(0.010322 \pm 0.007440\) |
| Macro mean | **\(0.017363\)** | \(0.006735\) | \(0.004367\) | \(0.008207\) |
| Average method rank \(\downarrow\) | **\(1.2\)** | \(2.6\) | \(3.4\) | \(2.8\) |

Full GenSHAP achieves the largest separation on four of the five held-out datasets and the best average rank. Its macro SeparationAUC is \(2.58\times\), \(3.98\times\), and \(2.12\times\) those of GIN-only, Transformer-only, and MLP-only, respectively.

The same pattern appears when considering removal of low-valued points alone: the macro \(\operatorname{AUC}_{\mathrm{low}}\) values are \(0.727807\), \(0.696328\), \(0.723472\), and \(0.724736\) for full GenSHAP, GIN-only, Transformer-only, and MLP-only, respectively. Full GenSHAP therefore gives the highest retained-model accuracy when progressively removing points predicted to be harmful or least useful.

### Conclusion from the extended ablation

The expanded evaluation gives a more nuanced result than MSE alone. MLP-only has the strongest pointwise MSE and global rank correlation, while Transformer-only is competitive in global ranking and extreme-set recovery. Full GenSHAP nevertheless gives the strongest performance on the metrics most directly connected to data-valuation use: it has the highest macro bottom-set recall at every evaluated cutoff, the highest ExtremeRecall at the two most selective cutoffs, and the highest downstream removal SeparationAUC on four of five datasets.

These findings suggest that the branches provide complementary information. The simpler ablations can better match individual Shapley coordinates on some datasets, whereas integrating graph structure, set-level interactions, and auxiliary statistics produces a more operationally useful valuation—particularly for recognizing harmful points and separating data whose removal has different effects on model utility. We will include these rank-recovery and removal-utility results alongside MSE so that the architectural comparison does not rely on a single metric.

## 9. Terminology and scope: completed answer

### Draft response to the AC, SaUX, and Z5Z8

We agree. GenSHAP does not compute exact Data Shapley values; it learns to predict finite-budget Permutation Data Shapley estimates generated with \(R=5000\). We will change the title to **“Towards Generalizable Data Valuation: Learning an End-to-End Model to Approximate Data Shapley Values”** and replace “compute,” “ground truth,” and unqualified “Shapley values” with “approximate,” “reference estimate,” and “Permutation Data Shapley estimate,” respectively, where appropriate.
We will also narrow “any unseen classification dataset” and “regardless of domain, sample size, or feature dimension” to the empirical scope supported here: source-disjoint held-out supervised classification tasks, using the same base learner/utility family and datasets of the tested scale. “Single forward pass” will refer only to the neural stage after deterministic target-dataset representation and auxiliary-feature construction.

Recommended abstract replacement:
GenSHAP learns an amortized predictor of high-budget Permutation Data Shapley estimates. After target-dataset representation and auxiliary-feature construction, the learned network predicts sample values with one neural forward pass. On five source-disjoint held-out classification sources under the evaluated LR setting, it closely tracks the permutation-based reference while reducing online valuation time.

For generative modeling, multitask learning, regression, unsupervised learning, or substantially shifted utility definitions, no supporting experiment exists. State these as future work rather than anticipated generalization.

## 10. KDE clarification

### Draft clarification to gcUt

We agree that the KDE objective was described inconsistently. The implementation uses \(w_i=\rho_i/\sum_j\rho_j\), so it gives greater weight to high-density target regions and reduces the influence of isolated extreme values. It does **not** upweight sparse tails. We will make the formula, motivation, and implementation description consistent and remove the contradictory “tail-sample neglect” wording.

The current density ablation is not rebuttal-ready: the no-density run used `Random_PCA`, while the full row used `Multi_V1`. It must be rerun on the same source-disjoint split before claiming a controlled benefit.


## 11. Monte Carlo variance of the Permutation-Shapley supervision
We thank the reviewers for raising this concern. Permutation Shapley with a finite number of permutations is a Monte Carlo estimator, so we explicitly quantified the variance introduced by using \(R=5000\) permutations and compared it with GenSHAP’s prediction error.
### Experimental protocol
For point \(i\), let \(\Delta_i^{(r)}\) be its marginal contribution in permutation \(r\). The reference value is
\[
\widehat{\phi}_i=\frac{1}{R}\sum_{r=1}^{R}\Delta_i^{(r)}, \qquad R=5000.
\]We saved the complete marginal-contribution vector from every permutation. The estimated variance of the \(R=5000\) reference for point \(i\) is
\[
\widehat{\mathrm{Var}}(\widehat{\phi}_i)
=
\frac{\widehat{\mathrm{Var}}(\Delta_i)}{R}.
\]Because GenSHAP is evaluated using MSE averaged over valued points, we define the dataset-level reference-noise floor as
\[
\mathrm{NoiseFloor}
=
\frac{1}{N}\sum_{i=1}^{N}
\widehat{\mathrm{Var}}(\widehat{\phi}_i).
\]This is the MSE expected even from a predictor of the latent, infinite-permutation Shapley values when it is evaluated against a finite-\(R\) reference.
We evaluated two representative held-out datasets selected to span substantially different regimes:
Iris: 75 valued points, four original features, and no PCA.
STL10: 964 valued points, an image-derived dataset represented by 20 PCA features.
For each dataset, we generated a new independent, untruncated \(R=5000\) reference using the same LR utility and held-out accuracy used for the original labels. We used three complementary validation procedures:
Block bootstrap: the 5,000 permutations were divided into 100 blocks of 50 permutations. Whole marginal-contribution vectors were resampled jointly using 2,000 bootstrap replicates.
Disjoint-half check: the first and second 2,500 permutations were treated as independent estimates. Since each half has twice the variance of an \(R=5000\) estimate, their MSE was divided by four.
Independent-reference check: the newly generated \(R=5000\) estimate was compared with the originally stored \(R=5000\) reference. Their MSE was divided by two because both estimates contain Monte Carlo variance.
Existing labels, model checkpoints, and predictions were read-only throughout this experiment.

| Dataset | Valued points | Analytical noise-floor MSE | Bootstrap noise-floor MSE | Bootstrap 95% interval | Disjoint-half check | Independent-reference check |
|---|---:|---:|---:|---:|---:|---:|
| Iris | 75 | 0.000001180563 | 0.000001155051 | [0.000001116279, 0.000001194889] | 0.000001137904 | 0.000001036411 |
| STL10 | 964 | 0.000000019772 | 0.000000019742 | [0.000000019431, 0.000000019859] | 0.000000019768 | 0.000000020126 |

The different estimates are highly consistent. For Iris, the bootstrap and disjoint-half estimates differ from the analytical estimate by only \(2.16\%\) and \(3.61\%\), respectively. The independent-reference check remains on the same scale and is \(12.21\%\) lower. For STL10, the bootstrap, half-split, and independent-reference estimates differ from the analytical estimate by only \(0.15\%\), \(0.02\%\), and \(1.79\%\), respectively.
We also verified Shapley efficiency for every generated permutation. The maximum absolute discrepancy between the sum of the marginal contributions and the full-data utility was \(5.13\times10^{-8}\) for Iris and \(3.05\times10^{-8}\) for STL10.

### Comparison with GenSHAP’s MSE
| Dataset | Submitted GenSHAP MSE | Analytical noise floor | MSE / noise floor | Noise share of observed MSE |
|---|---:|---:|---:|---:|
| Iris | 0.000181169744 | 0.000001180563 | 153.46 | 0.65% |
| STL10 | 0.000000526784 | 0.000000019772 | 26.64 | 3.75% |

For Iris, GenSHAP’s submitted MSE is approximately \(153\) times the estimated noise floor; reference-label uncertainty accounts for only \(0.65\%\) of the observed MSE. For STL10, the submitted MSE is approximately \(27\) times the noise floor, with reference uncertainty accounting for \(3.75\%\).
The conclusion is unchanged for the three newly repeated source-disjoint GenSHAP models:

| Dataset | Repeated-run MSE, mean \(\pm\) SD | MSE / noise floor | Noise share |
|---|---:|---:|---:|
| Iris | \(0.000232666398 \pm 0.000066361668\) | 197.08 | 0.51% |
| STL10 | \(0.000000524964 \pm 0.000000041807\) | 26.55 | 3.77% |

The predictions are also insensitive to which independent \(R=5000\) reference is used. For the submitted model, replacing the stored reference with the newly generated independent reference changes the measured MSE from \(0.000181169744\) to \(0.000185179304\) on Iris and from \(0.000000526784\) to \(0.000000535399\) on STL10. For the repeated source-disjoint models, the corresponding mean changes are only \(0.25\%\) and \(1.98\%\).

### Impact of Monte Carlo noise
Writing the finite-\(R\) reference as
\[
\widehat{\phi}=\phi+\epsilon,
\]where \(\phi\) is the latent infinite-permutation value and \(\epsilon\) is zero-mean Monte Carlo error, gives
\[
\mathbb{E}\!\left[(f-\widehat{\phi})^2\right]
=
\mathbb{E}\!\left[(f-\phi)^2\right]
+
\mathbb{E}\!\left[\epsilon^2\right]
\]when the held-out reference error is independent of the prediction. The source-disjoint GenSHAP models do not observe the test-task permutations or reference labels, so their predictions cannot reproduce the realization-specific Monte Carlo fluctuations in these references.
Subtracting the analytical floor gives approximate noise-corrected submitted MSEs of:
| Dataset | Observed MSE | Noise-corrected MSE | Fraction remaining |
|---|---:|---:|---:|
| Iris | 0.000181169744 | 0.000179989182 | 99.35% |
| STL10 | 0.000000526784 | 0.000000507012 | 96.25% |

Therefore, finite-\(R\) uncertainty contributes less than \(4\%\) of the observed error even in the more conservative of the two cases. GenSHAP’s MSE remains between \(26.64\) and \(153.46\) times the estimated reference-noise floor. Consequently, the reported results are not dominated by Monte Carlo variance, and GenSHAP has not reached—or merely fitted—the \(R=5000\) reference-noise floor.
This experiment does not claim that every dataset has identical Monte Carlo variance; the variance is inherently task-dependent. Instead, it quantifies the effect on two representative datasets spanning small/no-PCA and large/image-derived/PCA settings. The agreement among the analytical, bootstrap, split-half, and independent-reference estimates provides a direct and robust answer to the requested noise-floor diagnostic.



## 12. Empirical scalability beyond 1,000 valued points
We thank the reviewers for asking how GenSHAP behaves beyond the approximately 1,000-point regime used in the main accuracy experiments. We agree that the current implementation contains dense pairwise matrices, fully connected graphs, and full SVDs, and therefore should not be described as scalable to arbitrarily large \(N\). We performed an additional controlled benchmark to measure its practical behavior beyond the submitted regime.
### Experimental protocol
We loaded the frozen full GenSHAP checkpoint from the source-disjoint seed-23 run; no model was trained or updated in this experiment. We constructed target tasks with
\[
N\in\{250,500,1000,1500,2000,3000\}
\]valued points from the held-out STL10 task while holding the feature dimension and class structure fixed at 20 features and four balanced classes. For \(N\) below the original task size, points were sampled without replacement. For larger \(N\), we used class-balanced resampling with deterministic feature jitter of \(10^{-4}\) times the feature-wise standard deviation. This construction isolates computational scaling with \(N\); it is not used as a new valuation-accuracy experiment.
The timer includes the complete target-side online pipeline:
target CSV loading;
pairwise-distance and KNN-feature computation;
fitting the full LR utility model;
gradient-derived features;
the 21 learning-dynamics LR fits;
construction of five fully connected graphs;
five full SVD representations;
transfer of graph representations to the device;
and the final neural prediction.
Checkpoint loading was excluded from the online total and measured separately at \(0.053477\)–\(0.059834\) s. Reference Shapley values were not used. A zero-valued placeholder was supplied only because the current evaluation loader expects a PerDShap column; this attribute is not a model input and does not affect preprocessing or prediction.
The experiment was conducted on the same dual-socket Intel Xeon E5-2678 v3 server with 251 GiB system memory and one NVIDIA RTX 3090. CPU numerical libraries and PyTorch used four threads. Each \(N\) was evaluated in an isolated process using one untimed warm-up followed by three timed repetitions. Process isolation ensured that GPU memory was released between sizes and that peak-memory measurements were specific to each \(N\).

| \(N\) | Pairwise distance (s) | Auxiliary features and I/O (s) | Five full SVDs (s) | Other graph construction (s) | Neural forward (s) | Online total (s) | Peak CPU RSS (GiB) | Peak GPU allocation (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | \(0.001402 \pm 0.000105\) | \(0.121638 \pm 0.001468\) | \(0.059373 \pm 0.000054\) | \(0.011195 \pm 0.000108\) | \(0.019968 \pm 0.000518\) | \(0.213577 \pm 0.002013\) | 0.921 | 0.029 |
| 500 | \(0.003380 \pm 0.000328\) | \(0.143380 \pm 0.001210\) | \(0.229459 \pm 0.001341\) | \(0.034713 \pm 0.000150\) | \(0.025013 \pm 0.000408\) | \(0.435946 \pm 0.002651\) | 0.942 | 0.054 |
| 1,000 | \(0.009907 \pm 0.001607\) | \(0.197960 \pm 0.001279\) | \(1.271079 \pm 0.006595\) | \(0.112965 \pm 0.002063\) | \(0.032687 \pm 0.000548\) | \(1.624600 \pm 0.009877\) | 1.029 | 0.156 |
| 1,500 | \(0.019803 \pm 0.002310\) | \(0.218601 \pm 0.003970\) | \(3.696556 \pm 0.008662\) | \(0.329426 \pm 0.007418\) | \(0.043364 \pm 0.000812\) | \(4.307750 \pm 0.014568\) | 1.146 | 0.309 |
| 2,000 | \(0.035087 \pm 0.001549\) | \(0.277090 \pm 0.003762\) | \(8.304149 \pm 0.022000\) | \(0.572828 \pm 0.001021\) | \(0.059828 \pm 0.000709\) | \(9.248982 \pm 0.028600\) | 1.315 | 0.548 |
| 3,000 | \(0.088870 \pm 0.001954\) | \(0.391913 \pm 0.002034\) | \(28.504422 \pm 0.062140\) | \(1.439272 \pm 0.002668\) | \(0.094419 \pm 0.000525\) | \(30.518897 \pm 0.061576\) | 1.633 | 1.176 |

All entries are means \(\pm\) sample standard deviations over three timed repetitions. Every requested size completed successfully without timeout or out-of-memory failure.
The variation between repeated measurements is small. The coefficient of variation of the online total decreases from \(0.94\%\) at \(N=250\) to \(0.20\%\) at \(N=3000\), indicating stable timing measurements.
### Scaling behavior
The experiment directly shows that the current full GenSHAP implementation remains operational beyond the submitted \(N\approx1000\) regime:
At \(N=1500\), the complete online time is \(4.307750\) s.
At \(N=2000\), it is \(9.248982\) s.
At \(N=3000\), it is \(30.518897\) s.
Peak GPU allocation at \(N=3000\) is \(1.175926\) GiB, while peak CPU RSS is \(1.632854\) GiB.
However, the results also confirm the reviewers’ concern that full SVD becomes the limiting component. At \(N=1000\), the five SVDs account for \(78.24\%\) of the online total. This increases to \(89.78\%\) at \(N=2000\) and \(93.40\%\) at \(N=3000\). At the largest setting, the complete preprocessing and representation stage takes \(30.424477\) s, whereas the neural forward pass requires only \(0.094419\) s, or \(0.31\%\) of the online total.
Across the complete \(N=250\)–\(3000\) range, log-log regression gives empirical slopes of:
| Quantity | Empirical slope with respect to \(N\) |
|---|---:|
| Full-SVD time | 2.48 |
| Complete preprocessing time | 2.04 |
| Online-total time | 2.00 |
| Peak GPU allocation | 1.50 |

These finite-range slopes include substantial fixed costs at small \(N\). Restricting the fit to \(N\geq1000\) gives slopes of 2.83 for full-SVD time, 2.68 for online-total time, and 1.85 for peak GPU allocation. This high-\(N\) behavior approaches the expected cubic-time and quadratic-memory limitations of dense full SVD and graph construction.
Increasing \(N\) from 1,000 to 3,000 increases:
full-SVD time by \(22.43\times\);
complete online time by \(18.79\times\);
peak GPU allocation by \(7.54\times\);
and the number of stored directed graph edges from 4,995,000 to 44,985,000 across the five graph scales.

### Randomized-SVD acceleration and fidelity study
Because the preceding profile identifies full SVD as the dominant \(N=3000\) cost, we additionally tested whether a rank-100 randomized SVD could approximate the five spectral graph representations more efficiently. This is a controlled representation study on the same \(N=3000\), 20-feature, four-class STL10-derived target used above; it is not an additional valuation-accuracy result. We constructed the same five dense Gaussian-kernel matrices with \(c\in\{0.25,0.5,1,2,4\}\), and compared NumPy's full SVD with randomized rank-100 SVD using 20 oversampling vectors and \(q\in\{1,2,4,7\}\) power iterations. We used a fixed random seed, one untimed warm-up, and three timed repetitions. Only the decomposition was timed, so distance and kernel construction were excluded from this table.

We assessed approximation fidelity at two levels. First, we compared the singular values, left/right singular subspaces, feature Gram matrices, and rank-100 reconstruction errors, using the same 16 fixed Rademacher probes for every method. Second, we loaded the frozen exact-SVD GenSHAP checkpoint and changed only its five spectral node-feature matrices. We then measured agreement with the exact-SVD predictions using Pearson and Spearman correlation and overlap among the highest- and lowest-valued 10% of points. No model was retrained or selected in this experiment. Since the enlarged \(N=3000\) target has no reference Shapley labels, the prediction columns measure agreement with the exact-SVD pipeline rather than accuracy against ground truth.

| Representation | Five decompositions (s) | Speedup over matched full-SVD control | Singular-value relative \(L_2\) error | Feature-Gram relative error | Reconstruction error (exact / approximate) | Prediction Pearson | Prediction Spearman | Top / bottom 10% overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full SVD | \(84.278728 \pm 9.837241\) | \(1.00\times\) | 0 | 0 | \(0.136167 / 0.136167\) | 1.000000 | 1.000000 | \(1.000 / 1.000\) |
| Randomized, \(q=1\) | \(0.590311 \pm 0.008262\) | \(142.77\times\) | 0.008908 | 0.088872 | \(0.136167 / 0.143913\) | 0.653243 | 0.579792 | \(0.263 / 0.557\) |
| Randomized, \(q=2\) | \(0.708045 \pm 0.003950\) | \(119.03\times\) | 0.027342 | 0.096690 | \(0.136167 / 0.166316\) | 0.369244 | 0.341209 | \(0.177 / 0.300\) |
| Randomized, \(q=4\) | \(1.192136 \pm 0.005896\) | \(70.70\times\) | 0.000673 | 0.040943 | \(0.136167 / 0.137129\) | 0.843842 | 0.811581 | \(0.567 / 0.633\) |
| Randomized, \(q=7\) | \(1.710907 \pm 0.011235\) | \(49.26\times\) | 0.000177 | 0.029551 | \(0.136167 / 0.136575\) | 0.948677 | 0.930354 | \(0.777 / 0.850\) |

All representation errors are macro-averages over the five graph scales. The repeated full-SVD control in this later job was slower than the \(28.504422\) s measured in the isolated scaling benchmark above, indicating substantial between-job wall-clock variation on the shared server. We therefore retain the isolated measurements in the main scaling table rather than replacing them with this later control. The acceleration conclusion does not depend on using the slower control: relative to the earlier \(28.504422\) s full-SVD measurement, the \(q=4\) and \(q=7\) randomized variants are still \(23.91\times\) and \(16.66\times\) faster, respectively.

The higher-accuracy \(q=7\) setting provides a close rank-100 spectral approximation: its macro-averaged singular-value relative error is \(0.000177\), its left/right mean subspace cosines are \(0.995261/0.995581\), and its sketched reconstruction error increases only from \(0.136167\) to \(0.136575\). The remaining representation discrepancy is concentrated at the narrowest \(c=0.25\) graph: its feature-Gram relative error is \(0.144932\), whereas the corresponding error is at most \(0.001844\) for each of the other four scales. As an explicit component-substitution estimate, adding the measured \(1.710907\) s \(q=7\) decomposition time to the \(2.014475\) s non-SVD remainder from the earlier \(N=3000\) profile gives \(3.725382\) s, versus the measured \(30.518897\) s exact-SVD online total. This \(8.19\times\) figure is an engineering projection, not a newly measured end-to-end GenSHAP runtime.

Importantly, near-equivalent matrix reconstruction does not make randomized SVD a validated drop-in replacement for the frozen model. With \(q=7\), frozen-model prediction agreement reaches Pearson \(0.948677\) and Spearman \(0.930354\), but the relative prediction \(L_2\) error is \(0.201290\), and the top/bottom-decile overlaps are \(0.776667/0.850000\). These values do not meet our prespecified prediction-side drop-in checks of \(0.99\) Pearson and Spearman correlation, relative \(L_2\) error at most \(0.10\), and decile overlap at least \(0.90\). A plausible reason is that GenSHAP consumes the singular-vector coordinates directly: even when randomized SVD recovers almost the same low-rank matrix and subspace, small changes or rotations of those coordinates can affect a checkpoint trained exclusively on exact-SVD features.

We therefore draw a deliberately limited conclusion. Randomized rank-100 SVD is an empirically promising way to reduce the dominant spectral preprocessing cost, but the submitted exact-SVD checkpoint should not simply be switched to randomized features. A deployable randomized-SVD variant should be trained with the same randomized representation and then re-evaluated on source-disjoint held-out datasets. We have not performed that retraining here, so all accuracy and end-to-end runtime claims for the current method continue to refer to the exact/full-SVD implementation. Moreover, randomized SVD reduces the decomposition cost but does not remove the \(O(N^2)\) dense distance matrices and fully connected graphs.

### Interpretation and revised scope
These results answer the immediate empirical question of whether GenSHAP can operate beyond 1,000 valued points: on the tested hardware, the current implementation completes the entire online pipeline through \(N=3000\), with total runtime remaining below 31 seconds and peak measured GPU allocation below 1.2 GiB.
Nevertheless, this benchmark does not support a claim of unrestricted large-\(N\) scalability. The current implementation still uses five dense \(N\times N\) representations, fully connected graphs, and full SVD. Its asymptotic preprocessing cost remains
\[
O(BN^3)
\]
for \(B\) graph scales, with
\[
O(BN^2)
\]representation and graph memory. The favorable measured behavior through \(N=3000\) should therefore be interpreted as an empirical operating range, not as evidence that the implementation will remain practical for tens or hundreds of thousands of points.
We will revise the manuscript accordingly: GenSHAP is empirically practical through 3,000 valued points on the tested task and hardware, but dense graph and full-SVD construction remain its principal scalability limitation. The additional randomized-SVD study demonstrates substantial potential to accelerate the spectral component, while also showing that representation-consistent retraining is needed before claiming it as a replacement for the evaluated exact-SVD model. Sparse graph construction and Nyström-style representations remain complementary directions for extending the method to substantially larger datasets. We will distinguish these prospective or separately tested improvements from the current implementation rather than attributing their complexity or runtime to the method as presently evaluated.






## A. Manuscript corrections required regardless of new experiments

1. Replace the title’s “Compute Shapley Values” with “Approximate Data Shapley Values.”
2. Replace “100 datasets” with “100 constructed task pairs from 33 source datasets” for the main LR setting.
3. Replace Appendix Table 6 with the `Multi_V1` source manifest.
4. Rename the Random-PCA section so it does not claim strict cross-source generalization.
5. Correct Random-PCA GenSHAP Iris MSE from \(3.02\times10^{-5}\) to \(3.42\times10^{-4}\), unless a different traceable run is supplied.
6. Correct GenSHAP-S STL10 online time to approximately 1042.96 s for the logged main run.
7. Remove claims that the current mixed-split architecture, auxiliary, and density tables are controlled comparisons; rerun or delete the affected rows.
8. State that density-proportional KDE weights emphasize dense regions, not sparse tails.
9. State actual full-SVD \(O(BN^3)\) preprocessing and \(O(BN^2)\) memory; distinguish the separately tested randomized-SVD approximation from the evaluated exact-SVD implementation and do not present its projected runtime as measured end-to-end performance.
10. Narrow “any unseen dataset” and “regardless of domain, sample size, or feature dimension.”
11. Make “single forward pass” refer only to the neural prediction stage and report the full online decomposition.
12. Add a label-free inference path; `PerDShap` must be optional at deployment.
13. Repair and verify the anonymous repository link.

## B. Evidence anchors

- Main source-disjoint GenSHAP predictions: `Outputs_II/260415/Trial01/diagnostics_test_final`
- Main source-disjoint GenSHAP-S predictions: `Outputs_II/260415/Trial02/diagnostics_test_final`
- Repeated-partition GenSHAP runs, Splits 1--3: `Outputs_II/rebuttal_multiv2_source_disjoint_3splits_parallel`
- Repeated-partition GenSHAP run, Split 4: `Outputs_II/rebuttal_multiv2_additional_iris_heldout_seed20260727`
- Matched repeated-partition MLPbV predictions: `Outputs_II/rebuttal_mlpbv_source_disjoint_4splits`
- Main GenSHAP timing: `Outputs_II/260415/Trial01/Train_Test_Time.txt`
- GenSHAP-S complete post-label training timing: `Outputs_II/260415/Trial03/Train_Test_Time.txt`
- Main training corpus: `Datasets/Base_LR/Multi_V1/Train`
- Main held-out corpus: `Datasets/Base_LR/Multi_V1/Test`
- Random-PCA corpus: `Datasets/Base_LR/Random_PCA`
- Architecture scripts showing `Random_PCA`: `Scripts/Experiment_Plots/7_Ablation_MLP.py`, `7_Ablation_GIN.py`, and `7_Ablation_Tran.py`
- Permutation reference routine, which currently saves only the final average: `Scripts/True_DataShapley/Per_DShap_Multiclass_LR.py`
- Online preprocessing and timer: `Scripts/Model/GenSHAP_LR.py`
- \(N=3000\) exact-versus-randomized SVD timing, representation fidelity, and frozen-prediction agreement: `Outputs_II/rebuttal_exact_vs_randomized_svd_n3000`
