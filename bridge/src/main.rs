use axum::{routing::post, Router, Json, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;

// هيكل البيانات لمشاركة الحالة بين المسارات
struct AppState {
    tx: mpsc::Sender<String>, // قناة لإرسال النصوص للنواة
}

#[tokio::main]
async fn main() {
    // 1. إنشاء قناة التواصل (MPSC Channel)
    let (tx, mut rx) = mpsc::channel::<String>(100);
    let shared_state = Arc::new(AppState { tx });

    // 2. تشغيل النواة في Background Thread
    tokio::spawn(async move {
        while let Some(prompt) = rx.recv().await {
            println!("🧠 Core is processing: {}", prompt);
            // هنا يتم استدعاء ZumarBlock::forward من zumar-core
        }
    });

    // 3. إعداد الخادم
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("🚀 Zumar Bridge Linked to Core on port 8080");
    axum::serve(listener, app).await.unwrap();
}

async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    // إرسال النص للنواة عبر القناة
    let user_content = payload.messages.last().unwrap().content.clone();
    let _ = state.tx.send(user_content).await;

    Json(ChatResponse {
        id: "zumar-1".to_string(),
        content: "Success: Request sent to Zumar Core.".to_string(),
    })
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct Message {
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    content: String,
}
