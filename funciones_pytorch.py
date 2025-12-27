##### funciones para hacer fine tuning con pytorch con efficientnet_v2_s #########

######### funcion para hacer una matriz de confusion #########

def multi_conf_matrix(true_data, pred_data, classes, color="Blues"):
    """
    Plotea una matriz de confusión con valores absolutos y porcentajes.

    Args:
        true_data: Array con etiquetas verdaderas (numpy array o list).
        pred_data: Array con etiquetas predichas (numpy array o list).
        classes: Lista de nombres de clases (ej: ['Hitori', 'Nijika', 'Ryo', 'Kita']).
        color: Colormap para el heatmap (por defecto 'Blues').
    """
    cm = confusion_matrix(true_data, pred_data)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    # Manejo de casos donde alguna clase no aparece (evita NaN)
    cm_percentage = np.nan_to_num(cm_percentage)

    # Crear etiquetas con conteo absoluto y porcentaje
    labels = [f'{int(val)}\n{perc:.2f}%' for val, perc in zip(cm.flatten(), cm_percentage.flatten())]
    labels = np.asarray(labels).reshape(cm.shape)

    plt.figure(figsize=(10, 8))
    sn.heatmap(cm, annot=labels, fmt='', cmap=color, 
               xticklabels=classes, yticklabels=classes,
               cbar_kws={"label": "Count"})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    pass


###### matriz de confusion pero solo colorea los porcentajes #########

def multi_conf_matrix_2(true_data, pred_data, classes, color="Blues"):
    """
    Plotea una matriz de confusión coloreada según el porcentaje de acierto,
    sin números y con el mayor acierto en color más oscuro.

    Args:
        true_data: Array con etiquetas verdaderas.
        pred_data: Array con etiquetas predichas.
        classes: Lista de nombres de clases.
        color: Colormap base (se agregará _r para invertir). Por defecto 'Blues'.
    """
    cm = confusion_matrix(true_data, pred_data)
    
    # Porcentajes por fila
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percentage = np.nan_to_num(cm_percentage)

    # Invertir el colormap agregando '_r'
    inverted_color = color + "_r" if not color.endswith("_r") else color

    plt.figure(figsize=(10, 8))
    
    sn.heatmap(cm_percentage, 
               annot=False,                    # Sin números
               cmap=inverted_color,            # Colormap invertido → 100% más oscuro
               vmin=0, vmax=100,               # Escala fija 0-100%
               xticklabels=classes, 
               yticklabels=classes,
               cbar_kws={"label": "Porcentaje (%)"})
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusión - Mayor Acierto = Más Oscuro')
    plt.show()
    pass



###### funcion para crear el modelo de clasificacion de imagenes #########


def create_model(
    num_classes=4,   # Por defecto el que te recomiendo
    fine_tune_layers=0,
    extra_hidden_layers=0,            # Nuevo: número de capas Dense ocultas extra
    hidden_units=4                  # Nuevo: neuronas en cada capa oculta extra (si las hay)
):
    """
    Crea un modelo PyTorch basado en EfficientNetV2 (de torchvision).

    Args:
        num_classes (int): Número de clases de salida (por defecto 4).
        model_name (str): Solo "efficientnet_v2_s" por ahora (el mejor disponible nativo).
        fine_tune_layers (int): Número aproximado de capas superiores a descongelar (0 = todo congelado).
        extra_hidden_layers (int): Número de capas Dense ocultas adicionales antes de la salida (por defecto 0).
        hidden_units (int): Número de neuronas en cada capa oculta extra (por defecto 512).

    Returns:
        Modelo PyTorch listo para mover a device y entrenar.
    """
    try:
        # Cargar el modelo con pesos preentrenados de ImageNet
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        base_model = efficientnet_v2_s(weights=weights)

        # Congelar todas las capas inicialmente
        for param in base_model.parameters():
            param.requires_grad = False

        # Obtener el número de features de salida del backbone (después del pooling)
        in_features = base_model.classifier[1].in_features

        # Crear la nueva cabeza clasificadora
        layers = []

        # Capas ocultas extras (si el usuario quiere más de 0)
        current_in_features = in_features
        for i in range(extra_hidden_layers):
            layers.append(nn.Linear(current_in_features, hidden_units))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.5))  # Dropout común en fine-tuning
            current_in_features = hidden_units

        # Capa final de clasificación
        layers.append(nn.Linear(current_in_features, num_classes))

        # Reemplazar el classifier completo
        base_model.classifier = nn.Sequential(*layers)

        # Fine-tuning: descongelar las últimas capas si se indica
        if fine_tune_layers > 0:
            # Primero descongelamos todo
            for param in base_model.parameters():
                param.requires_grad = True

            # Luego volvemos a congelar todo excepto las últimas ~fine_tune_layers
            # (aproximamos contando módulos con parámetros hacia atrás)
            trainable_count = 0
            for module in reversed(list(base_model.modules())):
                if trainable_count >= fine_tune_layers:
                    for param in module.parameters(recurse=False):
                        param.requires_grad = False
                else:
                    params_in_module = sum(p.numel() for p in module.parameters(recurse=False))
                    if params_in_module > 0:
                        trainable_count += 1

            total_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in base_model.parameters())
            print(f"Fine-tuning activado: aproximadamente las últimas {fine_tune_layers} capas.")
            print(f"Parámetros entrenables: {total_trainable:,} / {total_params:,}")
        else:
            print("Todas las capas del backbone están congeladas (solo se entrena la cabeza).")

        return base_model

    except Exception as e:
        print(f"Error al crear el modelo: {e}")
        return None
    pass

######### early stopping #########


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience (int): Cuántos epochs esperar sin mejora antes de parar (igual que tu patience=3).
            min_delta (float): Mejora mínima considerada como progreso (0 = cualquier mejora).
            restore_best_weights (bool): Si True, al final carga los mejores pesos encontrados.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
        elif val_loss < self.best_loss - self.min_delta:
            # Mejora detectada
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict()
        else:
            # No hay mejora
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Early stopping activado")
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                    pass
                pass
            pass
        pass
    pass


######### para entrenar el modelo #########

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=50, patience=3, device='cuda'):
    """
    Entrena el modelo en PyTorch con validación, early stopping y registro de métricas.

    Args:
        model: Modelo PyTorch (ya movido a device o no).
        train_loader: DataLoader de entrenamiento.
        val_loader: DataLoader de validación.
        criterion: Función de pérdida (ej. nn.CrossEntropyLoss()).
        optimizer: Optimizador (ej. optim.Adam).
        num_epochs (int): Máximo número de epochs (por defecto 50).
        patience (int): Paciencia para EarlyStopping (por defecto 3).
        device (str): 'cuda' o 'cpu'.

    Returns:
        history: Diccionario con train_loss, train_acc, val_loss, val_acc por epoch.
    """
    model.to(device)  # Aseguramos que el modelo esté en el dispositivo correcto

    # EarlyStopping
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

    # History para gráficas
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print("Iniciando entrenamiento...\n")

    for epoch in range(num_epochs):
        # ------------------- ENTRENAMIENTO -------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # ------------------- VALIDACIÓN -------------------
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        # Guardar métricas
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Imprimir
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print("-" * 60)

        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping activado: no hay mejora en val_loss.")
            break

    print("Entrenamiento finalizado.")
    if early_stopping.best_loss is not None:
        print(f"Mejor val_loss: {early_stopping.best_loss:.4f}")

    return history


###### plotear el historial de entrenamiento #########
def plot_training_history(history, figsize=(12, 4)):
    """
    Grafica el historial de entrenamiento: Loss y Accuracy en train/val.

    Args:
        history (dict): Diccionario con claves 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
        figsize (tuple): Tamaño de la figura (ancho, alto). Por defecto (12, 4).
    """
    plt.figure(figsize=figsize)

    # Gráfica de Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Gráfica de Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', marker='o')
    plt.plot(history['val_acc'], label='Val Acc', marker='o')
    plt.title('Accuracy durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    pass

##### predecir imagen individual #########

def predict_and_visualize_image(image_path, model, class_names, device, img_height=224, img_width=224):
    """
    Predice la categoría de una imagen individual y la visualiza con título y probabilidades.

    Args:
        image_path (str): Ruta a la imagen (ej. '/content/anime/Hitori/001.jpg').
        model: Modelo PyTorch entrenado (ya en device).
        class_names (list): Lista de nombres de clases (ej. ['Hitori', 'Nijika', 'Ryo', 'Kita']).
        device: torch.device (cuda o cpu).
        img_height (int): Altura para redimensionar (default 224).
        img_width (int): Ancho para redimensionar (default 224).
    """
    try:
        # 1. Cargar y preparar la imagen (igual que en las transforms del dataset)
        img = Image.open(image_path).convert('RGB')
        original_img = img.copy()  # Guardar copia para mostrar (sin normalizar)

        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0)  # Añadir batch dimension: [1, 3, 224, 224]

        # 2. Mover a GPU y predecir
        model.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            outputs = model(img_tensor)              # Logits de salida
            probs = torch.softmax(outputs, dim=1)[0] # Probabilidades con softmax
            probs_percent = probs.cpu().numpy() * 100

        # 3. Obtener clase predicha y segunda más probable
        predicted_idx = torch.argmax(probs).item()
        predicted_class = class_names[predicted_idx]
        predicted_prob = probs_percent[predicted_idx]

        # Segunda clase
        top2_idx = torch.topk(probs, 2).indices.cpu().numpy()
        second_class = class_names[top2_idx[1]]
        second_prob = probs_percent[top2_idx[1]]

        # 4. Visualizar
        plt.figure(figsize=(6, 6))
        plt.imshow(original_img)
        plt.axis('off')
        plt.title(f'{predicted_class}', fontsize=16, pad=20)
        plt.suptitle(f'Más probable: {predicted_class} ({predicted_prob:.2f}%)\n'
                     f'Segunda más probable: {second_class} ({second_prob:.2f}%)',
                     fontsize=12, y=0.05, color='white', backgroundcolor='black', alpha=0.7)
        plt.show()

        # Opcional: imprimir también en consola
        print(f"Predicción: {predicted_class} ({predicted_prob:.2f}%)")
        print(f"Segunda opción: {second_class} ({second_prob:.2f}%)")

    except Exception as e:
        print(f"Error al procesar la imagen {image_path}: {e}")
        pass
    pass


##### predicciones aleatorias en validacion #########

def predict_and_visualize_random_samples(val_loader, model, class_names, num_samples=6, device='cuda'):
    """
    Muestra `num_samples` imágenes aleatorias del set de validación con sus predicciones.
    - Sin títulos en las imágenes.
    - Cuadro de texto debajo de cada imagen con buena separación.
    - Verde si acierto, rojo si error.

    Args:
        val_loader: DataLoader de validación.
        model: Modelo PyTorch entrenado (ya en device).
        class_names (list): Lista de nombres de clases.
        num_samples (int): Número de muestras aleatorias a mostrar (default 6).
        device: torch.device.
    """
    # Poner modelo en modo evaluación
    model.eval()

    # Recopilar una lista de imágenes y etiquetas verdaderas
    images = []
    true_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            images.append(imgs.cpu())       # Guardamos en CPU para mostrar después
            true_labels.append(labels.cpu())
            if len(images) * val_loader.batch_size >= 10000:  # Límite razonable
                break

    # Convertir a listas planas
    all_images = torch.cat(images, dim=0).numpy()           # [N, 3, 224, 224]
    all_labels = torch.cat(true_labels, dim=0).numpy()     # [N]

    total = len(all_images)
    if total < num_samples:
        print(f"Solo hay {total} imágenes en validación. Mostrando todas.")
        indices = list(range(total))
    else:
        indices = random.sample(range(total), num_samples)

    # Crear figura 2x3
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.ravel()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_np = all_images[idx]                        # [3, 224, 224]
            true_label = class_names[all_labels[idx]]

            # Preparar tensor para el modelo (añadir batch y mover a device)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)  # [1, 3, 224, 224]

            # Predicción
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            probs_percent = probs.cpu().numpy() * 100

            top_idx = np.argsort(probs_percent)[::-1]
            pred_class = class_names[top_idx[0]]
            pred_prob = probs_percent[top_idx[0]]
            second_class = class_names[top_idx[1]]
            second_prob = probs_percent[top_idx[1]]

            # Mostrar imagen (transponer a HWC y desnormalizar para visualización correcta)
            img_display = np.transpose(img_np, (1, 2, 0))    # [224, 224, 3]
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = std * img_display + mean          # Desnormalizar
            img_display = np.clip(img_display, 0, 1)
            img_display = (img_display * 255).astype(np.uint8)

            axes[i].imshow(img_display)
            axes[i].axis('off')

            # Texto debajo
            color = 'green' if pred_class == true_label else 'red'
            info_text = (
                f"Verdadera: {true_label}\n"
                f"1ª: {pred_class} ({pred_prob:.1f}%)\n"
                f"2ª: {second_class} ({second_prob:.1f}%)"
            )

            axes[i].text(
                0.5, -0.05, info_text,
                transform=axes[i].transAxes,
                fontsize=11, ha='center', va='top', color=color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95,
                          edgecolor=color, linewidth=2)
            )

    # Ajuste de espaciado (más espacio abajo para los textos)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.22, hspace=0.3, wspace=0.1)
    plt.show()
    pass



