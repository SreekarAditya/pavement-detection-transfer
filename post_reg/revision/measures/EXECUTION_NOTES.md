# Distributed clause execution

The independent SHACL validation tasks can run on two processor pods. Alternating task indices assign each task once, and the coordinator restores the original order before the committed summary and report functions execute. Both pods verify the sealed inputs and use the same committed validation function. Missing, duplicate or altered returned work stops execution.

The scheduling wrapper imports the committed measure module and calls its command-line function once. The module contains two command-line guards; executing both would repeat the run and fail when recreating the existing output directory. This is an entry-point correction only. The frozen measure calculations, inputs, eligibility rules, normalizations and reported denominators are unchanged.

Synthetic scheduling and end-to-end peer tests run before the study execution. Original tags and sealed analysis files remain unchanged.
