# Gates G4 to G6: rules fixed before any external image was processed

Protocol version 3.3.9. Written 15 November 2026, before pod session 2. Implemented in `pdt_s2_external.py` (preparation) and `pdt_s2_g4.py` (gates).

**Collections.** Seven are downloaded: Rome version 2, SVRDD Beijing, EcuPDD Ecuador, IRD Baghdad, HRP4K, Kent, PDD2026 Indonesia. Two fail G4 on the released label counts before any removal and are not downloaded: N-RDD2024 Turkey and Georgia (129 pothole boxes) and UAV-PDD2023 Tianjin (65 pothole boxes after keeping one of the three augmented copies of each crop).

**Pool.** Every released image with a label record (label file, possibly empty, or annotation entry); images without one are counted and left out. An image that cannot be fully decoded (a truncated file) is left out and listed; a partial decode would show the detector an image that was never captured. This rule was added on 15 November 2026 after the first session 2 run stopped on one truncated EcuPDD file (`EPDD/train/images/722724b7-imagen_8.jpg`, 64 KB); no detector had been run on any external image.

**Class mapping (frozen).** D00, D10, D20, D40 only from released classes that state longitudinal crack, transverse crack, alligator crack or pothole (or carry those codes): Rome Pothole → D40; SVRDD 0–3 → D00, D10, D20, D40; EcuPDD Pothole → D40, Longitudinal_Crack → D00, Transverse_Crack → D10, Alligator_Crack → D20; IRD 0–3 → D00, D10, D20, D40; HRP4K pothole → D40; Kent pothole → D40; PDD2026 Pothole → D40. Unmapped classes (for example Rome "Crack", EcuPDD Block_Crack and Patching, SVRDD patches and manholes, IRD SpeedBump, Kent "o") are not reference boxes. The archive's own class list is checked against the inventory where the archive has one.

**Geometry.** Image size after the EXIF orientation; normalized boxes scaled by it; polygons to their enclosing box (at least 3 points); clipped to the image; boxes under 1 pixel wide or high after clipping dropped; exact duplicates in an image dropped. Drops counted by class. Images with an EXIF orientation other than 1 are counted and over-represented in a label-only overlay sample of 12 images per collection, which a person views before the lock to confirm that boxes sit on the damage. A collection whose boxes are visibly misplaced in that sample is recorded and excluded before the lock.

**Overlap with RDD2022 (G4).** Reference: all 47,420 Kaggle RDD2022 images (train and validation from C0, test computed in session 2 with the same SSCD file). An external image is removed if linked by pHash distance 4 or fewer, or SSCD cosine 0.75 or more.

**Blocks (G5).** Within each collection after removal: links by pHash 4 or fewer or SSCD cosine 0.65 or more; connected components are blocks. No release provides verified road or video identifiers for all its images, so the same image-similarity rule is used for every collection. Limitation: frames of the same road that look different are not grouped, so block-bootstrap intervals may be too narrow.

**Eligibility (G4).** Four-class primary: at least 150 boxes in each of D00, D10, D20, D40 after removal, and at least 20 blocks. Pothole analysis: at least 150 pothole boxes and 20 blocks.

**G6.** At least 2 eligible collections from different releases and collection teams, for each analysis; otherwise the protocol's fallbacks (Section 7) apply.
