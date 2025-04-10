from typing import List, Optional

import requests
from bs4 import BeautifulSoup, Tag
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

router = APIRouter()

class ScraperRequest(BaseModel):
    url: str = Field(description="URL to scrape", examples=["https://example.com"])

class ScraperResponse(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    images: Optional[List[str]] = None

@router.post(
    "/scrape",
    response_model=List[ScraperResponse],
    summary="Scrape a webpage",
    description="Scrape a webpage for its title, description, keywords, and images.",
)
async def scrape(
    request: ScraperRequest = Body(..., example={"url": "https://example.com"})
) -> List[ScraperResponse]:
    try:
        response = requests.get(request.url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    soup = BeautifulSoup(response.content, "html.parser")

    title = soup.title.string if soup.title else None
    description = None
    keywords = None
    images = []

    for meta in soup.find_all("meta"):
        if not isinstance(meta, Tag):
            continue
        name = meta.get("name")
        if name == "description":
            content = meta.get("content")
            description = str(content) if content else None
        elif name == "keywords":
            content = meta.get("content")
            if content and isinstance(content, str):
                keywords = content.split(",")

    for img in soup.find_all("img"):
        if not isinstance(img, Tag):
            continue
        src = img.get("src")
        if src:
            images.append(str(src))

    return [
        ScraperResponse(
            title=title,
            description=description,
            keywords=keywords,
            images=images,
        )
    ]