#!/bin/bash

echo "Стартуем"

LAMBDAS=(0.0001 0.00001)

for L in "${LAMBDAS[@]}"
do
    echo "Testing Regularizer Lambda = $L"

    python3 train.py --reg_coef $L --steps 30000 --dim 2 --side 20
done

echo "конец"
