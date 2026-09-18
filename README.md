python generate_odd.py --text-file transcriptions.txt --ref-audio ai_vy.wav --ref-text ref_text.txt --output-dir gwen_dataset --device cuda:0 > odd.log 2>&1 & pid1=$!; python generate_even.py --text-file transcriptions.txt --ref-audio ai_vy.wav --ref-text ref_text.txt --output-dir gwen_dataset --device cuda:0 > even.log 2>&1 & pid2=$!; wait "$pid1"; s1=$?; wait "$pid2"; s2=$?; echo "Odd exit: $s1 | Even exit: $s2"

python generate_odd.py --text-file transcriptions.txt --ref-audio ai_vy.wav --ref-text ref_text.txt --output-dir gwen_dataset --device cuda:0 > odd.log 2>&1 & pid1=$!; python generate_even.py --text-file transcriptions.txt --ref-audio ai_vy.wav --ref-text ref_text.txt --output-dir gwen_dataset --device cuda:0 > even.log 2>&1 & pid2=$!; wait "$pid1"; s1=$?; wait "$pid2"; s2=$?; echo "Odd exit: $s1 | Even exit: $s2"

python3 generate_odd.py --text-file transcriptions.txt --ref-audio ai_vy.wav --ref-text ref_text.txt --output-dir gwen_dataset --device cuda:0 > odd.log 2>&1 &
pid1=$!

python3 generate_even.py --text-file transcriptions.txt --ref-audio ai_vy.wav --ref-text ref_text.txt --output-dir gwen_dataset --device cuda:0 > even.log 2>&1 &
pid2=$!

wait "$pid1"
s1=$?

wait "$pid2"
s2=$?

echo "Odd exit: $s1 | Even exit: $s2"
