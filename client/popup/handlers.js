import { MESSAGES } from "../scripts/constants/messages";
import { updateResult } from "./utils";

export const handleMessageResponse = (response, button, buttonText) => {
  if (buttonText) buttonText.textContent = "광고성 체크";
  if (button) button.disabled = false;

  const { predictions, probabilities } = response ?? {};

  if (!Array.isArray(predictions) || !Array.isArray(probabilities)) {
    return updateResult(MESSAGES.ERROR.POPUP);
  }

  const isPromotional = predictions[0];
  const probability = probabilities[0][0];

  if (isPromotional === 1) {
    return updateResult(
      MESSAGES.IS_PROMOTIONAL.YES,
      MESSAGES.IS_PROMOTIONAL.RATE(probability.toFixed(2))
    );
  }
  return updateResult(
    MESSAGES.IS_PROMOTIONAL.NO,
    MESSAGES.IS_PROMOTIONAL.RATE(probability.toFixed(2))
  );
};
