# Task 05C adjudication viewer

Before this change, the two answer exports had been checked only for item coverage and answer validity. Agreement had not been computed.

The adjudication viewer displays the two original answers as “First answer” and “Second answer,” including their categories and subclasses where recorded. It assigns display order per item using seed 2028 and saves that assignment privately for resolving the two agree buttons. Only top-level disagreements go to adjudication. Subclass matches among items both raters marked as missing mapped distress are reported separately as unadjudicated.

The viewer reuses the full images and crops from the rebuilt rating packet, including the magenta outlines and the corrected crop windows. The sample, definitions and rule document are unchanged.

Before comparison of the exports, the raters confirmed that one used the rebuilt magenta-outline viewer and one used the original yellow-outline viewer with the earlier crop behavior. The item identifiers are the same across versions. The rebuilt adjudication images reproduce the rebuilt viewer's rendering. Task 05C was paused before agreement was computed.

## Re-rating plan

The rater who used the original yellow-outline packet will re-rate the audited magenta-outline packet in its existing item order (seed 2027). The viewer starts with blank answers. She will work alone, without seeing her earlier answers or discussing items. Her re-rating and the other rater's existing magenta-packet export will form the primary pair for Task 05C. Only top-level disagreements in this pair go to magenta-packet adjudication.

The original yellow-packet export remains frozen with its checksum. In a separate rendering-effect analysis, pair that export with the same rater's new export by item and report observed agreement and Cohen's kappa over the five top-level categories. Record that the yellow rating came first and the magenta rating is a repeat assessment of the same items.

This plan was recorded before any comparison of the answers. All comparisons wait for the re-rating export and its recorded SHA-256.
