Currency Detection System
Overview
YOLO-based Detection: Detects all banknotes in an image and localizes them with bounding boxes.
CNN-based Classification: Each detected note is classified by country and denomination using an enhanced attention-based multi-task CNN.

goal
Detect and classify banknotes from multiple countries.
Predict both country and denomination accurately, even when multiple notes are present in a single image.

approach
Shared CNN backbone extracts features.
Two heads predict country and denomination separately.
Real-time detection and classification of banknotes and coins.
