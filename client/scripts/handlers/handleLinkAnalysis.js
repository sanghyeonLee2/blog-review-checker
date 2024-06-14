import { EVENT_ACTIONS } from "../constants/eventActions";

const handleLinkAnalysis = async (tabId, url) => {
  try {
    const response = await sendMessage(tabId, {
      action: EVENT_ACTIONS.FETCH_AT_OUTER,
      url,
    });
    console.log("페이지 제목: ", response);
  } catch (err) {
    console.error(err);
  }
};

const sendMessage = (tabId, message) =>
  new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, message, (...args) => {
      chrome.runtime.lastError
        ? reject(new Error(chrome.runtime.lastError.message))
        : resolve(...args);
    });
  });

export default handleLinkAnalysis;
