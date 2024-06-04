from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from app.services.ocr import get_ocr_text
import traceback

async def crawl_blog_content(url):

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="load")

            # iframe src 추출
            outer_html = await page.content()
            outer_soup = BeautifulSoup(outer_html, "html.parser")
            iframe_tag = outer_soup.find("iframe")
            iframe_src = iframe_tag.get("src") if iframe_tag else None

            if not iframe_src:
                return {"title": "", "content": "", "ocr_data": ""}

            if iframe_src.startswith("/"):
                iframe_src = "https://blog.naver.com" + iframe_src

            # iframe으로 이동
            await page.goto(iframe_src, wait_until="load")

            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            title = soup.find("h3", class_="se_textarea")
            content_tags = soup.select('span[class^="se-fs-"]')
            images = soup.select("img")

            title_text = title.get_text(strip=True) if title else ""
            content_text = "\n".join(tag.get_text(strip=True) for tag in content_tags)

            # 이미지 추출 (Selenium과 동일한 클래스 우선 순위)
            img_tags = soup.select('img[class$="egjs-visible"], img')

            valid_imgs = [
                img for img in img_tags
                if img.has_attr("src") and "postfiles.pstatic.net" in img["src"]
            ]

            image_url = valid_imgs[-1]["src"] if valid_imgs else None

            ocr_text = get_ocr_text(image_url) if image_url else ""

            await browser.close()

            return {
                "title": title_text,
                "content": content_text,
                "ocr_data": ocr_text
            }

    except Exception as e:
        print("예외 발생:", e)
        traceback.print_exc()
        return {"error": str(e)}
