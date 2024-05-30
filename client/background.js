chrome.runtime.onInstalled.addListener(() => {
  console.log("Extension installed");
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "fetchPageTitle",
      title: "홍보성 체크",
      contexts: ["link"],
    });
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== "fetchPageTitle") return;
  fetchPageTitle(tab.id, info.linkUrl);
});

const fetchPageTitle = async (tabId, url) => {
  try {
    await executeScriptAsync({
      target: { tabId },
      files: ["./scripts/content.js"],
    });

    const response = await sendMessageAsync(tabId, {
      action: "fetchPageTitle",
      url,
    });

    console.log("Page title:", response);
  } catch (err) {
    console.error(err);
  }
};

const executeScriptAsync = (details) =>
  new Promise((resolve, reject) => {
    chrome.scripting.executeScript(details, () => {
      chrome.runtime.lastError
        ? reject(new Error(chrome.runtime.lastError))
        : resolve();
    });
  });

const sendMessageAsync = (tabId, message) =>
  new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      chrome.runtime.lastError
        ? reject(new Error(chrome.runtime.lastError))
        : resolve(response);
    });
  });
