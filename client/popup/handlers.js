import { MESSAGES } from "../scripts/constants/messages";
import { updateResult } from "./utils";
import { parsePredictionResult } from "../scripts/utils/utils";

export const handleMessageResponse = (response, button, buttonText) => {
  if (buttonText) buttonText.textContent = "홍보성 체크";
  if (button) button.disabled = false;

  const result = parsePredictionResult(response);
  if (!result) {
    return updateResult(MESSAGES.ERROR.POPUP);
  }

  const { isPromotional, percent } = result;

  return updateResult(
    isPromotional ? MESSAGES.IS_PROMOTIONAL.YES : MESSAGES.IS_PROMOTIONAL.NO,
    `리뷰가 홍보성일 확률 ${percent}%`
  );
};
