python generate_odd.py \
  --text-file transcriptions.txt \
  --ref-audio ai_vy.wav \
  --ref-text ref_text.txt \
  --output-dir gwen_dataset \
  --device cuda:0 > odd.log 2>&1 &

python generate_even.py \
  --text-file transcriptions.txt \
  --ref-audio ai_vy.wav \
  --ref-text ref_text.txt \
  --output-dir gwen_dataset \
  --device cuda:0 > even.log 2>&1 &

wait
echo "Both processes finished!"
