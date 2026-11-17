# Proposed Next Steps

The current model is a baseline, not a finished collapse detector. The Equinox
table contains 98 labeled polity-century observations but only 6 cases meeting
the chosen 30% territory-loss rule. Results should therefore be treated as
descriptive and unstable.

## Priorities

1. **Improve the data first.** Find more trustworthy historical observations of
   major territory loss and document every added source, join, and uncertainty.
   Do not fill missing values merely to increase the sample size.

2. **Check the label definition.** A 30% land loss is an operational rule, not
   automatically a collapse. Compare it with CrisisDB crisis records and test
   alternative, pre-specified land-loss rules. Report how conclusions change;
   do not choose a rule because it produces better scores.

3. **Investigate false alarms.** Review the five current high-score false
   positives against historical notes and missingness. Record whether each is a
   genuine survival, a label artifact, a data gap, or evidence that the model
   lacks an important predictor.

4. **Test unseen states.** Use polity-held-out evaluation and, where possible,
   a genuinely external historical set. A warning threshold can be increased
   as a decision trade-off, but fewer false alarms may also mean missed real
   contractions. Threshold changes do not by themselves improve the model.

## Current status

Completed work includes the Equinox-only baseline, grouped and temporal leakage
checks, OOF predictions, coefficient explanations, false-positive export, and a
Streamlit dashboard. The four priorities above have **not** been completed.

## Viva wording

> Our priority is more reliable data, especially more actual land-loss cases.
> We would check our definition against historical crisis records, investigate
> the false alarms, and test improvements on unseen states. These are proposed
> next steps, not improvements we've already completed.