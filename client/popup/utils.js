import { BLOG_DOM_QUERY_SELECTORS } from "../scripts/constants/domSelectors";

export const updateResult = (mainText, subText = "") => {
  const resultElement = document.getElementById(
    BLOG_DOM_QUERY_SELECTORS.ID.RESULT
  );
  const rateResultElement = document.getElementById(
    BLOG_DOM_QUERY_SELECTORS.ID.RATE_RESULT
  );

  if (resultElement) resultElement.textContent = mainText;
  if (rateResultElement) rateResultElement.textContent = subText;
};
