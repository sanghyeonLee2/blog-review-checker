chrome.runtime.onInstalled.addListener(() => {
  console.log("Extension installed");
  createContextMenu();
});

const createContextMenu = () => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "fetchPageTitle",
      title: "홍보성 체크",
      contexts: ["link"],
    });
  });
};

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== "fetchPageTitle") return;
  handleLinkAnalysis(tab.id, info.linkUrl);
});

const handleLinkAnalysis = async (tabId, url) => {
  try {
    await runScript(tabId, "./scripts/content.js");

    const response = await sendMessage(tabId, {
      action: "fetchPageTitle",
      url,
    });

    console.log("Page title:", response);
  } catch (err) {
    console.error("Error during link analysis:", err);
  }
};

const wrapChromeCallback = (fn) =>
  new Promise((resolve, reject) => {
    fn((...args) => {
      chrome.runtime.lastError
        ? reject(new Error(chrome.runtime.lastError.message))
        : resolve(...args);
    });
  });

const runScript = (tabId, file) =>
  wrapChromeCallback((cb) =>
    chrome.scripting.executeScript({ target: { tabId }, files: [file] }, cb)
  );

const sendMessage = (tabId, message) =>
  wrapChromeCallback((cb) => chrome.tabs.sendMessage(tabId, message, cb));
