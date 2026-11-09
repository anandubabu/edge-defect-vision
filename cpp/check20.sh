#!/bin/bash
# Run the C++ classifier on the first 20 test images and compare with the true class
correct=0
total=0
tail -n +2 ../data/split/test.csv | head -20 | while IFS=, read -r path label cls; do
    pred=$(./build/classify ../models/defect_int8_ov.xml "../$path" | cut -d' ' -f1)
    total=$((total + 1))
    if [ "$pred" == "$cls" ]; then
        correct=$((correct + 1))
        echo "OK     $cls"
    else
        echo "WRONG  true=$cls  predicted=$pred"
    fi
    echo "$correct / $total" > /tmp/score.txt
done
echo "correct: $(cat /tmp/score.txt)"