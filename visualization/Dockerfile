FROM python:3.11.8

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    streamlit \
    polars \
    pyarrow \
    plotly \
    folium \
    streamlit-folium

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "-m", "streamlit", "run", "visualization.py", "--server.address", "0.0.0.0"]