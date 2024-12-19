import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

async function runInference() {
    const MODEL_DIR = 'assets/model';
    const TOKENS_FILE = 'assets/model/token_and_mask.json';

    const tokensData = await fetch(TOKENS_FILE).then(res => res.json());
    const inputIds = tokensData.input_ids;
    const attentionMask = tokensData.attention_mask;

    // Set backend to WebGPU for performance
    await tf.setBackend('webgpu');
    await tf.ready();
    console.log(tf.backend());

    const status = document.getElementById('status');
    const output = document.getElementById('results');

    status.textContent = 'Loading model...';

    const model = await tf.loadGraphModel(`${MODEL_DIR}/model.json`);
    status.textContent = 'Model loaded!';

    const inputTensor = tf.tensor([inputIds], [1, inputIds.length], 'int32');
    const attentionMaskTensor = tf.tensor([attentionMask], [1, attentionMask.length], 'int32');

    status.textContent = 'Starting 30-minute loop...';
    const endTime = Date.now() + 1000 * 60 * 30;

    while (Date.now() < endTime) {
        const outputTensor = await model.executeAsync({
            input_ids: inputTensor,
            attention_mask: attentionMaskTensor,
        });
        outputTensor.forEach(t => t.dispose());
    }

    status.textContent = '30-minute loop completed. Running additional iterations...';

    const executionTimes = [];

    for (let i = 0; i < 100; i++) {
        const startTime = performance.now();

        const outputTensor = await model.executeAsync({
            input_ids: inputTensor,
            attention_mask: attentionMaskTensor,
        });

        for (let i = 0; i < outputTensor.length ; i++){
            await outputTensor[i].data()
        }

        const executionTime = performance.now() - startTime;
        executionTimes.push(executionTime);

        outputTensor.forEach(t => t.dispose());

        const paragraph = document.createElement('p');
        paragraph.textContent = `Iteration ${i + 1}: Execution time = ${executionTime.toFixed(2)} ms`;
        output.appendChild(paragraph);
    }

    status.textContent = 'All iterations completed!';
}



runInference().catch(err => {
    document.getElementById('status').textContent = `Error: ${err.message}`;
    console.error(err);
});
