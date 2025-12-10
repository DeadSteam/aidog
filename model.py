"""
Vision Transformer (ViT) модель для классификации собак Stanford Dogs Dataset
"""
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTModel, ViTConfig
from pathlib import Path


class VisionTransformer(nn.Module):
    """
    Vision Transformer модель для классификации
    """
    def __init__(self, num_classes=120, pretrained=True, image_size=224, patch_size=16):
        """
        Args:
            num_classes: Количество классов для классификации
            pretrained: Использовать ли предобученные веса
            image_size: Размер входного изображения
            patch_size: Размер патча
        """
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Загрузка предобученной модели или создание новой
        if pretrained:
            # Используем предобученную ViT из Hugging Face
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
            config = self.vit.config
        else:
            # Создаем ViT с нуля
            config = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
            )
            self.vit = ViTModel(config)
        
        # Замена классификационной головы
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
        # Инициализация классификатора
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Входной тензор изображений [batch_size, 3, image_size, image_size]
        
        Returns:
            Логиты для каждого класса [batch_size, num_classes]
        """
        outputs = self.vit(pixel_values=x)
        # Используем [CLS] токен для классификации
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits


class DogDataset(torch.utils.data.Dataset):
    """
    Датасет для Stanford Dogs Dataset
    """
    def __init__(self, images_dir, annotations_dir=None, transform=None, split='train'):
        """
        Args:
            images_dir: Директория с изображениями
            annotations_dir: Директория с аннотациями (не используется, но оставлена для совместимости)
            transform: Трансформации для изображений
            split: Сплит датасета ('train', 'val', 'test')
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.split = split
        
        # Сбор всех изображений и меток
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Получение списка классов
        classes = sorted([d.name for d in self.images_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Сбор всех изображений
        for class_name in classes:
            class_dir = self.images_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Загружено {len(self.images)} изображений из {len(classes)} классов для {split}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Загрузка изображения
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки изображения {img_path}: {e}")
            # Возвращаем пустое изображение в случае ошибки
            image = Image.new('RGB', (224, 224), color='black')
        
        # Применение трансформаций
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train', image_size=224):
    """
    Получить трансформации для датасета
    
    Args:
        split: 'train', 'val' или 'test'
        image_size: Размер изображения
    
    Returns:
        transforms.Compose объект
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # val или test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

