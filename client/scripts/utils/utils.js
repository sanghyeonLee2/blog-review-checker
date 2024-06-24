export const parsePredictionResult = (result) => {
  const { predictions, probabilities } = result ?? {};
  if (!Array.isArray(predictions) || !Array.isArray(probabilities)) {
    return null;
  }

  const isPromotional = predictions[0];
  const probability = probabilities[0][0]; // 홍보성 확률이라고 가정
  const percent = (isPromotional === 1 ? probability : 1 - probability) * 100;

  return { isPromotional, percent: percent.toFixed(2) };
};
