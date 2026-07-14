## Gold standard prompts
Note that the idea behind these gold standard prompts is to have a known completion, not a device specific completion.
In this case the known completion is based on prompting with the first part of two common license texts that likely
appear many times in training datasets, and then we verify that a model can complete at least the next paragraph of the
license. Note that the paragraph break seems to be the last point where we get identity out of some models (the first
instance of `\n\n`).

Please do not change the gold standard results for a100/h100 for this test without carefully considering if the result
is still "correct". These are not arbitrary outputs conditional on a device, they are specific outputs based on a common
text that should be overrepresented in training so should be easy for a relatively competent model to complete exactly.