import { fetchFromOuter } from "../api/apis";
import { MESSAGES } from "../constants/messages";
import displayResult from "../ui/displayResult";

const handleOuterContentCheck = async (request, sendResponse) => {
  try {
    const result = await fetchFromOuter(request.url);
    displayResult(result, request.url);
    sendResponse(result);
  } catch (err) {
    console.error(MESSAGES.ERROR.HANDLE_OUTER_CHECK, err);
    sendResponse({ error: err.message });
  }
};

export default handleOuterContentCheck;
