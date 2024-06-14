import { API_URLS } from "../constants/urls";
import { MESSAGES } from "./../constants/messages";

const postRequest = async (endPoint, data, timeout = 5000) => {
  const url = `${API_URLS.dev}${endPoint}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`${MESSAGES.HTTP} ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(MESSAGES.ERROR.REQUEST("POST", url), error);
    throw error;
  }
};

export default postRequest;
