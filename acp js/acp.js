const fs = require('fs');
const { EigenvalueDecomposition } = require('ml-matrix');
const PCA = require('ml-pca').PCA;

leerCSV();

async function leerCSV() {
    //start time
    const startTime = Date.now();
    const data = await fs.promises.readFile('EjemploEstudiantes.csv', 'utf-8');
    const rows = data.split('\n');
    const matriz = [];

    for (let i = 1; i < rows.length - 1; i++) {
        const values = rows[i].split(';').slice(1); // elimina el primer valor de la fila
        const row = values.map(value => {
            if (value.includes(',')) {
                return parseFloat(value.replace(',', '.'));
            } else {
                return parseFloat(value);
            }
        });
        matriz.push(row);
    }
    console.log("******** Tabla de datos *********")
    console.log(matriz);

    /* ------------------ CALCULO DE MATRIZ DE CORRELACIONES ---------------------*/
    // CALCULAR PROMEDIO DE COLUMNAS EN UN ARREGLO   
    const avgs = calcAvg(matriz);
    console.log("\n-. Cálculo de promedios de columnas");
    console.log(avgs);

    // CALCULAR DESVIACIÓN ESTANDAR DE COLUMNAS 
    const desvStd = calcDesviacionStd(matriz, avgs);
    console.log("\n-. Cálculo de desviación estándar de columnas");
    console.log(desvStd);

    // 1. CALCULO DE MATRIZ NORMALIZADA
    const matrizNormalizada = normalizarMatriz(matriz, avgs, desvStd);
    console.log("\n1. Cálculo de matriz Normalizada");
    printMatriz(matrizNormalizada)

    // 2. CALCULO DE MATRIZ DE CORRELACION
    const matrizCorrelacion = calcMatrizCorrelacion(matrizNormalizada);
    console.log("\n2. Cálculo de matriz de correlación");
    printMatriz(matrizCorrelacion);

    // 3 y 4. CÁLCULO DE VALORES Y VECTORES PROPIOS
    const resultado = calcVectoresValoresPropios(matrizCorrelacion);
    const valoresPropios = resultado.realEigenvalues;
    const vectoresPropios = resultado.eigenvectorMatrix.to2DArray();
    ordenarPropios(valoresPropios, vectoresPropios);
    console.log("\n3. Cálculo de valores propios ordenados");
    console.log(valoresPropios);
    console.log("\n4. Cálculo de vectores propios ordenados")
    printMatriz(vectoresPropios);

    // 5. CALCULAR LOS COMPONENTES PRINCIPALES
    const pca = new PCA(matrizNormalizada);
    const pc = pca.predict(matrizNormalizada).to2DArray(); // usando librería de machine learning
    //const pc = calcMatrizPC(matrizNormalizada, vectoresPropios);
    console.log("\n5. Cálculo de componentes principales");
    printMatriz(pc);

    // 6. ENCONTRAR LA MATRIZ DE CALIDAD DE INDIVIDUOS
    const matrizCalidadDeIndividuos = calcMatrizCalidadesDeIndividuos(pc, matrizNormalizada);
    console.log("\n6. Matriz de calidad de individuos");
    printMatriz(matrizCalidadDeIndividuos);

    // 7. ENCONTRAR LA MATRIZ DE COORDENADAS
    const matrizCorrelacionCoordenadas = calcMatrizCorrelacion(pc);
    const res = calcVectoresValoresPropios(matrizCorrelacionCoordenadas);
    const valoresPropiosCoordenadas = res.realEigenvalues;
    const vectoresPropiosCoordenadas = res.eigenvectorMatrix.to2DArray();
    const matrizCoordenadas = calcMatrizCoordenadas(pc, vectoresPropiosCoordenadas);
    console.log("\n7. Matriz de coordenadas de variables");
    printMatriz(matrizCoordenadas);

    //8. ENCONTRAR LA MATRIZ DE CALIDADES
    const matrizCalidades = calcMatrizCalidades(vectoresPropios, matrizCoordenadas);
    console.log("\n8. Matriz de Calidad");
    printMatriz(matrizCalidades);

    // 9. ENCONTRAR VARIANZA EXPLICADA PARA ENCONTRAR EL NÚMERO DE COMPONENTES PRINCIPALES
    const expVariance = explainedVariance(valoresPropios);
    console.log("\n9. Vector de Inercia: Porcentaje de varianza explicada por cada componente");
    console.log(expVariance);
    const endTime = Date.now();
    const timeElapsed = endTime - startTime; // in milliseconds
    console.log("Tiempo de ejecucion: " + timeElapsed)
}

function calcAvg(matriz) {
    const array = matriz[0].slice(0);
    for (let i = 1; i < matriz.length; i++) {
        for (let j = 0; j < array.length; j++) {
            array[j] += matriz[i][j];
        }
    }
    for (let i = 0; i < array.length; i++) {
        array[i] /= matriz.length;
    }
    return array;
}

function calcDesviacionStd(matriz, avgs) {
    const desvStd = [];
    const avgDiferencias = [];
    const n = matriz.length;
    for (let i = 0; i < n; i++) { // cada fila
        for (let j = 0; j < avgs.length; j++) { // cada celda
            const difference = matriz[i][j] - avgs[j];
            const squaredDiff = difference ** 2;
            if (i === 0) {
                avgDiferencias[j] = squaredDiff;
            } else {
                avgDiferencias[j] += squaredDiff;
            }
            if (i === (n - 1)) {
                desvStd[j] = Math.sqrt(avgDiferencias[j] / n);
            }
        }
    }
    return desvStd;
}

function normalizarMatriz(matriz, avgs, desvStd) {
    const matrizNormalizada = [];
    const nMat = matriz.length;
    const nAvg = avgs.length;
    for (let i = 0; i < nMat; i++) { // cada fila
        const row = [];
        for (let j = 0; j < nAvg; j++) {
            const z = (matriz[i][j] - avgs[j]) / desvStd[j];
            row.push(z);
        }
        matrizNormalizada.push(row);
    }
    return matrizNormalizada;
}

function printMatriz(matriz) {
    for (let i = 0; i < matriz.length; i++) {
        let row = matriz[i];
        let fila = row.map(num => num.toFixed(15)).join('\t'); // Use '\t' to separate elements with a tab
        console.log(fila);
    }
}

function calcMatrizCorrelacion(matrizNormalizada) {
    const matrizCorrelacion = [];
    const n = matrizNormalizada[0].length;
    for (let i = 0; i < n; i++) {
        const row = [];
        const col1 = getColumn(matrizNormalizada, i);
        for (let j = 0; j < n; j++) {
            // valor r de correlacion para la matriz 
            const col2 = getColumn(matrizNormalizada, j);
            const r = calcularValorCorrelacion(col1, col2);
            row.push(r);
        }
        matrizCorrelacion.push(row);
    }
    return matrizCorrelacion;
}

function getColumn(matrizNormalizada, i) {
    const col = [];
    const n = matrizNormalizada.length;
    for (let j = 0; j < n; j++) {
        col.push(matrizNormalizada[j][i]);
    }
    return col;
}

function calcularValorCorrelacion(col1, col2) {
    const n = col1.length;
    let sumaDiffx2 = 0; // sumatoria de diferencias cuadradas
    let sumaDiffy2 = 0;
    let numerador = 0;

    // cálculo de promedios de X y Y
    let promedioCol1 = 0;
    let promedioCol2 = 0;
    for (let i = 0; i < n; i++) {
        promedioCol1 += col1[i];
        promedioCol2 += col2[i];
    }
    promedioCol1 /= n;
    promedioCol2 /= n;

    // diferencias
    for (let i = 0; i < n; i++) {
        const diffx = col1[i] - promedioCol1;
        const diffy = col2[i] - promedioCol2;
        numerador += diffx * diffy;

        // sumatoria de diferencias cuadradas
        sumaDiffx2 += diffx ** 2;
        sumaDiffy2 += diffy ** 2;
    }

    const r = numerador / (Math.sqrt(sumaDiffx2) * Math.sqrt(sumaDiffy2));
    return r;
}

function calcVectoresValoresPropios(matriz) {
    const resultado = new EigenvalueDecomposition(matriz);
    return resultado;
}

// Ordena los valores propios de manera descendiente con sus vectores propios
function ordenarPropios(valoresPropios, vectoresPropios) {
    const n = valoresPropios.length;
    for (let i = 0; i < n; i++) {
        let mayor = valoresPropios[i];
        let index = -1;
        for (let j = i + 1; j < n; j++) {
            if (mayor < valoresPropios[j]) {
                mayor = valoresPropios[j];
                index = j;
            }
        }
        if (index != -1) {
            valoresPropios[index] = valoresPropios[i];
            valoresPropios[i] = mayor;
            // swap columnas del vector
            for (let j = 0; j < n; j++) {
                const temp = vectoresPropios[j][index];
                vectoresPropios[j][index] = vectoresPropios[j][i];
                vectoresPropios[j][i] = temp;
            }
        }
    }
}

function explainedVariance(valoresPropios) {
    const n = valoresPropios.length;
    const expVariance = [];
    let suma = valoresPropios[0];
    for (let i = 1; i < n; i++) {
        suma += valoresPropios[i];
    }
    for (let i = 0; i < n; i++) {
        expVariance.push((valoresPropios[i] / suma) * 100);
    }
    return expVariance;
}

// Calcular la matriz de componentes principales
function calcMatrizPC(matrizNormalizada, vectoresPropios) {
    const pc = [];
    const nNormalizada = matrizNormalizada.length;
    const nVectores = vectoresPropios[0].length;
    for (let i = 0; i < nNormalizada; i++) {
        const row = [];
        for (let j = 0; j < nVectores; j++) {
            row.push(productoPunto(matrizNormalizada[i], columnToArray(vectoresPropios, j)));
        }
        pc.push(row);
    }
    return pc;
}

function productoPunto(vec1, vec2) {
    const n = vec1.length;
    let suma = 0;
    for (let i = 0; i < n; i++) {
        suma += vec1[i] * vec2[i];
    }
    return suma;
}

function columnToArray(matriz, index) {
    const array = [];
    const n = matriz.length;
    for (let i = 0; i < n; i++) {
        array.push(matriz[i][index]);
    }
    return array;
}

function calcMatrizCalidadesDeIndividuos(pc, matrizNormalizada) {
    const matrizCalidadesDeIndividuos = [];
    const nFilas = pc.length;
    const nCols = pc[0].length;
    for (let i = 0; i < nFilas; i++) {
        const row = [];
        for (let j = 0; j < nCols; j++) {
            const numerador = pc[i][j] ** 2;
            let denominador = 0;
            for (let k = 0; k < nCols; k++) {
                denominador += matrizNormalizada[i][k] ** 2;
            }
            row.push(numerador / denominador);
        }
        matrizCalidadesDeIndividuos.push(row);
    }
    return matrizCalidadesDeIndividuos;
}

function calcMatrizCoordenadas(pc, vectoresPropiosCoordenadas) {
    const matrizCoordenadas = [];
    const nFilas = pc.length;
    const nCols = pc[0].length;
    for (let i = 0; i < nFilas; i++) {
        const row = [];
        for (let j = 0; j < nCols; j++) {
            row.push(productoPunto(pc[i], columnToArray(vectoresPropiosCoordenadas, j)));
        }
        matrizCoordenadas.push(row);
    }
    return matrizCoordenadas;
}

function calcMatrizCalidades(vectoresPropios, matrizCoordenadas) {
    const matrizCalidades = [];
    const nFilas = matrizCoordenadas.length;
    const nCols = vectoresPropios.length;
    const n = vectoresPropios[0].length;
    for (let i = 0; i < nCols; i++) {
        const row = [];
        for (let j = 0; j < nCols; j++) {
            let suma = 0;
            for (let k = 0; k < n; k++) {
                suma += matrizCoordenadas[i][k] * vectoresPropios[j][k];
            }
            row.push(suma);
        }
        matrizCalidades.push(row);
    }
    return matrizCalidades;
}