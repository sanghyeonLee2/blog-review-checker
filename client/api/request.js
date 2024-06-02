import { API_BASE_URL } from "../constants/env.js";

export const postRequest = async (endPoint, data) => {
  const url = `${API_BASE_URL}${endPoint}`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  return response.json();
};
