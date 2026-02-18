/**
 * MongoDB 聊天相关集合初始化脚本
 *
 * 使用方式：
 *   mongosh "<your-mongo-uri>/<db-name>" resources/mongodb/init_chat_collections.js
 *
 * 说明：
 * 1. 会幂等创建/更新 conversations、admin_messages 的 JSON Schema 校验；
 * 2. 会创建所需索引（重复执行不会报错）。
 */

const conversationsValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["uuid", "conversation_type", "user_id", "create_time", "update_time"],
    properties: {
      uuid: {
        bsonType: "string",
        description: "会话业务唯一ID"
      },
      conversation_type: {
        enum: ["client", "admin"],
        description: "会话类型"
      },
      user_id: {
        bsonType: "long",
        description: "用户ID（int64）"
      },
      title: {
        bsonType: "string",
        description: "会话标题"
      },
      create_time: {
        bsonType: "date",
        description: "创建时间"
      },
      update_time: {
        bsonType: "date",
        description: "更新时间"
      },
      message_count: {
        bsonType: "int",
        minimum: 0,
        description: "消息总数"
      },
      metadata: {
        bsonType: "object",
        description: "扩展信息"
      }
    }
  }
};

const adminMessagesValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["uuid", "conversation_id", "role", "content", "created_at", "updated_at"],
    properties: {
      uuid: {
        bsonType: "string",
        description: "消息业务唯一ID（UUID）"
      },
      conversation_id: {
        bsonType: "objectId",
        description: "所属会话ID"
      },
      role: {
        enum: ["user", "assistant"],
        description: "消息角色"
      },
      content: {
        bsonType: "string",
        description: "消息内容"
      },
      thought_chain: {
        bsonType: ["array", "null"],
        description: "思维链结构"
      },
      token_usage: {
        bsonType: ["object", "null"],
        description: "LLM Token 消耗明细",
        properties: {
          prompt_tokens: {
            bsonType: "int",
            minimum: 0,
            description: "输入 Token 数"
          },
          completion_tokens: {
            bsonType: "int",
            minimum: 0,
            description: "输出 Token 数"
          },
          total_tokens: {
            bsonType: "int",
            minimum: 0,
            description: "总 Token 数"
          },
          breakdown: {
            bsonType: ["array", "null"],
            description: "节点级 Token 明细",
            items: {
              bsonType: "object",
              required: ["node_name", "model_name", "prompt_tokens", "completion_tokens", "total_tokens"],
              properties: {
                node_name: {
                  bsonType: "string",
                  description: "节点名称"
                },
                model_name: {
                  bsonType: "string",
                  description: "模型名称"
                },
                prompt_tokens: {
                  bsonType: "int",
                  minimum: 0,
                  description: "输入 Token 数"
                },
                completion_tokens: {
                  bsonType: "int",
                  minimum: 0,
                  description: "输出 Token 数"
                },
                total_tokens: {
                  bsonType: "int",
                  minimum: 0,
                  description: "节点总 Token 数"
                }
              }
            }
          }
        }
      },
      created_at: {
        bsonType: "date",
        description: "创建时间"
      },
      updated_at: {
        bsonType: "date",
        description: "更新时间"
      }
    }
  }
};

function ensureCollectionWithValidator(collectionName, validator) {
  const exists = db.getCollectionInfos({ name: collectionName }).length > 0;
  if (!exists) {
    db.createCollection(collectionName, { validator });
    print(`Created collection: ${collectionName}`);
    return;
  }

  db.runCommand({
    collMod: collectionName,
    validator: validator,
    validationLevel: "strict",
    validationAction: "error"
  });
  print(`Updated validator for collection: ${collectionName}`);
}

// ==============================
// conversations
// ==============================
ensureCollectionWithValidator("conversations", conversationsValidator);

db.conversations.createIndex({ uuid: 1 }, { unique: true });
db.conversations.createIndex({ user_id: 1, conversation_type: 1, update_time: -1 });

// ==============================
// admin_messages
// ==============================
ensureCollectionWithValidator("admin_messages", adminMessagesValidator);

db.admin_messages.createIndex({ uuid: 1 }, { unique: true });
db.admin_messages.createIndex({ conversation_id: 1, created_at: 1 });
db.admin_messages.createIndex({ conversation_id: 1, created_at: -1 });
db.admin_messages.createIndex({ conversation_id: 1, "token_usage.total_tokens": -1 });

print("MongoDB chat collections initialized.");
