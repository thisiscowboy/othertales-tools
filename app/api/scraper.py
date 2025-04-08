from typing import List, Optional

import requests
from bs4 import BeautifulSoup, Tag
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

class ScraperResponse(BaseModel):
    title: str
    description: str
    url: str

@router.get("/scrape", response_model=List[ScraperResponse])
async def scrape_website(url: str):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    soup = BeautifulSoup(response.content, "html.parser")
    # Handle potential None value for title
    title = soup.title.string if soup.title else "No title"
    
    # Properly handle the meta description
    description_tag = soup.find("meta", {"name": "description"})
    description = "No description"
    if description_tag and isinstance(description_tag, Tag):
        content = description_tag.get("content")
        if content:
            description = content

    return [ScraperResponse(title=title or "No title", description=description, url=url)]