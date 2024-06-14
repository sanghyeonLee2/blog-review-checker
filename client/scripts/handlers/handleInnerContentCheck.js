import { fetchFromInner } from "../api/apis";
import { BLOG_DOM_QUERY_SELECTORS } from "../constants/domSelectors";
import { MESSAGES } from "../constants/messages";

const handleInnerContentCheck = async (_, sendResponse) => {
  const iframe = document.querySelector(BLOG_DOM_QUERY_SELECTORS.IFRAME);
  if (!iframe) {
    sendResponse(MESSAGES.ERROR.NOT_FOUND_IFRAME);
    return;
  }

  try {
    const doc = iframe.contentDocument || iframe.contentWindow.document;

    const textBlocks = doc.querySelectorAll(BLOG_DOM_QUERY_SELECTORS.TEXT);
    const contentText = [...textBlocks].map((el) => el.innerText).join("\n");

    const imgs = doc.querySelectorAll(BLOG_DOM_QUERY_SELECTORS.IMG);
    const imageUrl = imgs.length > 0 ? imgs[imgs.length - 1].src : "";

    const titleElement = doc.querySelector(BLOG_DOM_QUERY_SELECTORS.TITLE);
    const title = titleElement ? titleElement.innerText : "";

    const result = await fetchFromInner({ title, contentText, imageUrl });
    sendResponse(result);
  } catch (err) {
    console.error(MESSAGES.ERROR.HANDLE_INNER_CHECK, err);
    sendResponse({ error: err.message });
  }
};
export default handleInnerContentCheck;
