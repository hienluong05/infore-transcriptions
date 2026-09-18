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

Nen tai torch ver nao?
2026-09-18 15:57:49,215 INFO Shard=even | Source: transcriptions.txt | utterances: 7567 | ref: ai_vy.wav (10.3s) | output: /u01/user-data/sontc/hienluong/generate-dataset-omnivoice-main/generate-dataset-omnivoice-main/infore-transcriptions-main/gwen_dataset
2026-09-18 15:57:49,216 INFO Example: {'id': '000002_001', 'line': 2, 'part': 1, 'text': 'Cảm ơn quý khách đã liên hệ với trung tâm chăm sóc khách hàng của chúng tôi.'}
Traceback (most recent call last):
  File "/u01/user-data/sontc/hienluong/generate-dataset-omnivoice-main/generate-dataset-omnivoice-main/infore-transcriptions-main/generate_even.py", line 271, in <module>
    main()
  File "/u01/user-data/sontc/hienluong/generate-dataset-omnivoice-main/generate-dataset-omnivoice-main/infore-transcriptions-main/generate_even.py", line 206, in main
    import torch
ModuleNotFoundError: No module named 'torch'

Tai thanh cong chua
<img width="1582" height="600" alt="image" src="https://github.com/user-attachments/assets/357f4b66-137b-4163-984f-ad14b09a577d" />

<img width="1582" height="34" alt="image" src="https://github.com/user-attachments/assets/4995cd4f-c5f1-4ac7-88c1-fb0e7fb62170" />

