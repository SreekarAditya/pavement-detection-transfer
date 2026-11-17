# Task 05B packet rendering correction

The first packet build clipped crop windows at image boundaries and reduced large crops to a 640-pixel maximum. Of 450 crops, 293 were trimmed at an image edge, reduced by the cap, or both. Seventy had a side shorter than 160 pixels. The image audit also flagged two full images for possible extra outlines. Packet delivery was held, and no ratings had been collected.

The replacement slides each crop window within the source image and preserves its source pixels. Each axis uses twice the alert box size, rounded up, with a minimum of 160 pixels where the source permits. Both views outline the same alert in magenta. The window bounds stay in the private key.

The replacement uses the verified Part A archive and retains the sampling code, seeds, item identities, and rater order. The frozen rule files are unchanged. The rebuilt packets must pass the full independent audit before delivery.
