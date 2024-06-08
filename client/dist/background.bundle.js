// scripts/constants/eventActions.js
var EVENT_ACTIONS = {
  FETCH_AT_OUTER: "fetchAtOuter",
  FETCH_AT_INNER: "fetchAtInner"
};

// scripts/handlers/handleLinkAnalysis.js
var handleLinkAnalysis = async (tabId, url) => {
  try {
    const response = await sendMessage(tabId, {
      action: EVENT_ACTIONS.FETCH_AT_OUTER,
      url
    });
    console.log("\uD398\uC774\uC9C0 \uC81C\uBAA9: ", response);
  } catch (err) {
    console.error(err);
  }
};
var sendMessage = (tabId, message) => new Promise((resolve, reject) => {
  chrome.tabs.sendMessage(tabId, message, (...args) => {
    chrome.runtime.lastError ? reject(new Error(chrome.runtime.lastError.message)) : resolve(...args);
  });
});
var handleLinkAnalysis_default = handleLinkAnalysis;

// background/background.js
var OPTION_TITLE = "\uAD11\uACE0\uC131 \uCCB4\uD06C";
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: EVENT_ACTIONS.FETCH_AT_OUTER,
      title: OPTION_TITLE,
      contexts: ["link"]
    });
  });
});
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== EVENT_ACTIONS.FETCH_AT_OUTER) return;
  handleLinkAnalysis_default(tab.id, info.linkUrl);
});
