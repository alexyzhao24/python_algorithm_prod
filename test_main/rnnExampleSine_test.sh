## Generating all data
python rnnExampleSine.py --gen_data

## Training all models with full-cycle data and save
python rnnExampleSine.py --model_choice gru --train_data full
python rnnExampleSine.py --model_choice lstm --train_data full
python rnnExampleSine.py --model_choice rnn --train_data full

## Training all models with half-cycle data and save
python rnnExampleSine.py --model_choice gru --train_data half
python rnnExampleSine.py --model_choice lstm --train_data half
python rnnExampleSine.py --model_choice rnn --train_data half

## Testing all models and save
python rnnExampleSine.py --test_only --train_data full

python rnnExampleSine.py --test_only --train_data half


